# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import re
import mmcv
from mmcv.utils import Config
from deris.models import build_model
from deris.utils import load_checkpoint
from deris.datasets import extract_data
from mmcv.parallel import collate
import os
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.colors as mplc
from transformers import XLMRobertaTokenizer
from deris.datasets.pipelines.transforms import HierResize, Normalize
from deris.datasets.pipelines.formatting import DefaultFormatBundle, CollectData
import matplotlib as mpl
from deris.utils.visualizer import GenericMask, VisImage


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--img",
        default="tools/imgs/COCO_train2014_000000197503the_little_boy_with_the_orange_shirt_img.jpg",
        help="Image file",
    )
    parser.add_argument("--expression", default="the little boy with the orange shirt", help="text")
    parser.add_argument(
        "--config",
        default="configs/gres/sota/gref_sota_swinbase384_beitlarge224_ep6_nq20_mt50_rate0.15_mint0_simthr_0.6.py",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/gres/sota/gref_sota_swinbase384_beitlarge224_ep3_nq20_mt50_rate0.15_mint0_simthr_0.6/20250227_114627/lightweight_model/compressed.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--output_dir", default="asserts/outdir", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.9, help="bbox score threshold")
    args = parser.parse_args()
    return args


class DERIS:
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.cfg.img = args.img
        self.cfg.expression = args.expression
        self.cfg.output_dir = args.output_dir
        self.cfg.device = args.device
        self.cfg.model["post_params"]["score_threshold"] = args.score_thr
        self.model = build_model(self.cfg.model)
        self.model.to(args.device)
        load_checkpoint(self.model, load_from=args.checkpoint)
        self.max_token = 50
        self.tokenizer = XLMRobertaTokenizer("pretrain_weights/beit3.spm")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        file_client_cfg = dict(backend="disk")
        self.file_client = mmcv.FileClient(**file_client_cfg)
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.Trans_HierResize = HierResize(
            img_scale=(384, 384),
            mini_img_scale=(224, 224),
            keep_ratio=False,
        )
        self.Trans_Normalize = Normalize(**img_norm_cfg)
        self.Trans_DefaultFormatBundle = DefaultFormatBundle()
        self.Trans_CollectData = CollectData(
            keys=[
                "img",
                "img_mini",
                "ref_expr_inds",
                "text_attention_mask",
            ],
            meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "empty"],
        )

    def clean_string(self, expression):
        return re.sub(r"([.,'!?\"()*#:;])", "", expression.lower()).replace("-", " ").replace("/", " ")

    def inference_detector(self):
        results = {}
        img, text = self.cfg.img, self.cfg.expression
        img_bytes = self.file_client.get(img)
        image = mmcv.imfrombytes(img_bytes, flag="color", backend=None)
        results["filename"] = img
        results["img"] = image
        results["img_shape"] = image.shape  # (h, w, 3), rgb default
        results["ori_shape"] = image.shape
        results["empty"] = None
        expression = self.clean_string(text)
        tokens = self.tokenizer.tokenize(expression)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if len(tokens) > self.max_token - 2:
            tokens = tokens[: self.max_token - 2]
        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_token - num_tokens)
        ref_expr_inds = tokens + [self.pad_token_id] * (self.max_token - num_tokens)
        results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
        results["text_attention_mask"] = np.array(padding_mask, dtype=int)
        results["expression"] = expression
        results["max_token"] = self.max_token
        results["with_bbox"] = False
        results["with_mask"] = False
        result_resize = self.Trans_HierResize(results)
        result_normalize = self.Trans_Normalize(result_resize)
        result_defaultFormatBundle = self.Trans_DefaultFormatBundle(result_normalize)
        result_CollectData = self.Trans_CollectData(result_defaultFormatBundle)
        data = collate([result_CollectData], samples_per_gpu=1)
        inputs = extract_data(data)
        img_metas = inputs["img_metas"]

        predictions = self.model(**inputs, return_loss=False, rescale=True, with_bbox=True, with_mask=True)

        pred_masks = predictions.pop("pred_masks")
        for j, (img_meta, pred_mask) in enumerate(zip(img_metas, pred_masks)):
            filename, expression = img_meta["filename"], img_meta["expression"]
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            outfile = os.path.join(self.cfg.output_dir, expression.replace(" ", "_") + "_" + os.path.basename(filename))
            self.imshow_expr_mask(filename, pred_mask["pred_masks"], outfile)

    def imshow_expr_mask(self, filename, pred_mask, outfile):
        img = cv2.imread(filename)[:, :, ::-1]
        height, width = img.shape[:2]
        img = np.ascontiguousarray(img).clip(0, 255).astype(np.uint8)
        output_pred = VisImage(img, scale=1.0)
        pred_mask = maskUtils.decode(pred_mask)
        assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
        pred_mask = GenericMask(pred_mask, height, width)
        for segment in pred_mask.polygons:
            polygon = mpl.patches.Polygon(
                segment.reshape(-1, 2),
                fill=True,
                facecolor=mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,),
                edgecolor=mplc.to_rgb([0.0, 0.0, 0.0]) + (1,),
                linewidth=2,
            )
            output_pred.ax.add_patch(polygon)
        cv2.imwrite(outfile.replace(".jpg", "_pred.jpg"), output_pred.get_image()[:, :, ::-1])

    def forward(self):
        self.inference_detector()


if __name__ == "__main__":
    args = parse_args()
    Demo = DERIS(args)
    Demo.forward()
