import copy
from matplotlib.colors import ListedColormap
from torch.nn import functional as F
import torch
import numpy
from deris.core.structure.boxes import Boxes
from deris.core.structure.instances import Instances
from deris.core.structure.postprocessing import detector_postprocess
from deris.layers.box_ops import box_cxcywh_to_xyxy
from deris.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils

from deris.models.branchs.perception_branch.mask_config.config import get_mask_config
from deris.models.builder import build_branch
from deris.models.postprocess.nms import nms
import numpy as np
from deris.utils import is_main


from deris.models import MODELS, build_head
from .base import BaseModel
from .base import custom_colors as colors
from torch import nn
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageOps
import cv2


@MODELS.register_module()
class MIXGrefUniModel_HierVG_MRLoopback(BaseModel):
    def __init__(
        self,
        understanding_branch,  # {vis_encoder, understanding_config}
        perception_branch,  # {vis_encoder, mask_config}
        head,
        hidden_dims=256,
        mask_save_target_dir=None,
        visualize_params={"post_paramsenable": True, "row_columns": (4, 5), "train_interval": 50, "val_interval": 5},
        post_params={
            "score_weighted": True,
            "mask_threshold": 0.5,
            "score_threshold": 0.7,
            "with_nms": False,
            "outmask_type": "global",
        },
        vis_box=False,
        vis_mask=True,
        visual_mode="val",
        **kwargs,
    ):
        super(MIXGrefUniModel_HierVG_MRLoopback, self).__init__()
        self.threshold = post_params["mask_threshold"]
        self.score_threshold = post_params["score_threshold"]
        self.score_weighted = post_params["score_weighted"]
        self.with_nms = post_params["with_nms"]
        self.visualize_params = visualize_params
        self.vis_box = vis_box
        self.vis_mask = vis_mask
        self.outmask_type = post_params.get("outmask_type", None)
        self.nt2 = post_params.get("nt2", False)
        self.enable_visualize = False
        if is_main() and visualize_params["enable"] and mask_save_target_dir is not None:
            self.train_mask_save_target_dir = os.path.join(mask_save_target_dir, "train_vis")
            self.val_mask_save_target_dir = os.path.join(mask_save_target_dir, "val_vis")
            self.test_mask_save_target_dir = os.path.join(mask_save_target_dir, "test_vis")
            os.makedirs(self.train_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.val_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.test_mask_save_target_dir, exist_ok=True)
            self.enable_visualize = True
        self.understanding_branch = build_branch(understanding_branch)
        self.perception_branch = build_branch(perception_branch)
        self.visual_mode = visual_mode
        self.head = build_head(head)
        self.iter = 0

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        gt_bbox=None,
        gt_mask_rle=None,
        gt_mask_parts_rle=None,
        rescale=False,
        epoch=None,
        img_mini=None,
    ):
        # TODO 设定query
        feat_dict = self.understanding_branch.pre_forward(img_mini, ref_expr_inds, text_attention_mask)
        query_feat = feat_dict["query_feat"]
        img_feat = feat_dict["img_feat"]
        text_feat = feat_dict["text_feat"]
        cls_feat = feat_dict["cls_feat"]

        perception_results = self.perception_branch(img, img_feat, text_feat, text_attention_mask, query_feat)
        perception_queries = perception_results["query_feat"]

        understanding_results = self.understanding_branch.post_forward(
            perception_queries,
            perception_results,
            feat_dict["img_feat"],
            feat_dict["text_feat"],
            text_attention_mask,
            cls_feat,
        )

        if understanding_results["pred_global_mask"] is not None:
            pred_global_mask = F.interpolate(
                understanding_results["pred_global_mask"],
                size=perception_results["pred_masks"].shape[-2:],
                mode="bilinear",
            )
        else:
            pred_global_mask = None

        pred_dict = {
            "pred_boxes": perception_results["pred_boxes"],  # (B,N,4)
            "pred_masks": perception_results["pred_masks"],  # (B,N,H,W)
            "pred_logits": perception_results["pred_refer_logits"],  # (B,N,2)
            "pred_existence": understanding_results["pred_existence"],  # (B,1)
            "pred_global_mask": pred_global_mask,  # 上采样到swin的输入
            "aux_outputs": perception_results["aux_outputs_refer"],
        }

        perception_pred_dict = {
            "pred_boxes": perception_results["pred_boxes"],  # (B,N,4)
            "pred_masks": perception_results["pred_masks"],  # (B,N,H,W)
            "pred_logits": perception_results["pred_logits"],  # (B,N,2)
            "aux_outputs": perception_results["aux_outputs_perception"],
        }

        targets = {
            "mask": gt_mask_rle,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "epoch": epoch,
            "mask_parts": gt_mask_parts_rle,
        }
        losses_dict = self.head.forward_train(
            predictions=pred_dict, perception_prediction=perception_pred_dict, targets=targets
        )

        with torch.no_grad():
            predictions = self.get_predictions_parts(
                pred_dict, img_metas, rescale=rescale, with_bbox=True, with_mask=True
            )
        self.iter += 1
        extra_dict = {}
        if is_main() and self.iter % self.visualize_params["val_interval"] == 0 and self.enable_visualize:
            try:
                self.visualiation_parts(
                    predictions["parts_list"],
                    img_metas,
                    targets,
                    self.train_mask_save_target_dir,
                    extra_dict,
                    text_attention_mask=text_attention_mask,
                )
            except Exception as e:
                print(e)

        return losses_dict, None

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        with_bbox=False,
        with_mask=False,
        gt_bbox=None,
        gt_mask=None,
        gt_mask_parts_rle=None,
        rescale=False,
        img_mini=None,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `rec/datasets/pipelines/formatting.py:CollectData`.

        with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
            which has slight differences.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        # TODO 设定query
        feat_dict = self.understanding_branch.pre_forward(img_mini, ref_expr_inds, text_attention_mask)
        query_feat = feat_dict["query_feat"]
        img_feat = feat_dict["img_feat"]
        text_feat = feat_dict["text_feat"]
        cls_feat = feat_dict["cls_feat"]

        perception_results = self.perception_branch(img, img_feat, text_feat, text_attention_mask, query_feat)
        perception_queries = perception_results["query_feat"]

        understanding_results = self.understanding_branch.post_forward(
            perception_queries,
            perception_results,
            feat_dict["img_feat"],
            feat_dict["text_feat"],
            text_attention_mask,
            cls_feat,
        )

        if understanding_results["pred_global_mask"] is not None:
            pred_global_mask = F.interpolate(
                understanding_results["pred_global_mask"],
                size=perception_results["pred_masks"].shape[-2:],
                mode="bilinear",
            )
        else:
            pred_global_mask = None

        pred_dict = {
            "pred_boxes": perception_results["pred_boxes"],  # (B,N,4)
            "pred_masks": perception_results["pred_masks"],  # (B,N,H,W)
            "pred_logits": perception_results["pred_refer_logits"],  # (B,N,2)
            "pred_existence": understanding_results["pred_existence"],  # (B,1)
            "pred_global_mask": pred_global_mask,  # 上采样到swin的输入
        }

        predictions = self.get_predictions_parts(
            pred_dict,
            img_metas,
            rescale=rescale,
            with_bbox=with_bbox,
            with_mask=with_mask,
        )
        predictions["instance"] = perception_results["pred_masks"]
        targets = {
            "mask": gt_mask,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "mask_parts": gt_mask_parts_rle,
        }

        self.iter += 1
        extra_dict = {}

        # if is_main() and self.iter % self.visualize_params["val_interval"] == 0 and self.enable_visualize:
        if is_main() and self.iter % 2 == 0 and self.enable_visualize:
            try:
                self.visualiation_parts(
                    predictions["parts_list"],
                    img_metas,
                    targets,
                    self.val_mask_save_target_dir if self.visual_mode == "val" else self.test_mask_save_target_dir,
                    extra_dict,
                    text_attention_mask=text_attention_mask,
                )
            except Exception as e:
                print(e)

        return predictions

    def get_predictions_parts(self, pred, img_metas, rescale=False, with_bbox=False, with_mask=False):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """
        pred_bboxes, pred_masks = [], []
        pred_logits, pred_bboxes_, nt_labels, pred_mask_, global_seg_mask_ = (
            pred.get("pred_logits", None),
            pred.get("pred_boxes", None),
            pred.get("pred_existence", None),
            pred.get("pred_masks", None),
            pred.get("pred_global_mask", None),
        )
        # refer_pred = refer_pred.sigmoid()
        scores, nms_indices = [], []
        if self.nt2:
            nt_labels = nt_labels.softmax(-1)
            nt_labels = nt_labels[:, 1].unsqueeze(1)
        else:
            nt_labels = nt_labels.sigmoid()
        parts_list = {
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
            "pred_global_mask": [],
            "att_map": [],
            "nt": [],
        }

        bboxes = pred_bboxes_
        bbox_cls = pred_logits

        if bboxes is not None and with_bbox:
            image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
            results = self.inference(bbox_cls, bboxes, image_sizes)

            for ind, (results_per_image, img_meta) in enumerate(zip(results, img_metas)):
                pred_nt = 1 - nt_labels[ind]
                image_size = img_meta["img_shape"]
                height = image_size[0]
                width = image_size[1]
                r = detector_postprocess(results_per_image, height, width)
                # infomation extract
                pred_box = r.pred_boxes.tensor
                refer_score = r.scores
                score = refer_score
                if self.score_weighted:
                    score = score * (1 - pred_nt[0])
                    # score = score * 0.8 + (1 - pred_nt[0]) * 0.2
                    # score = torch.max(score, (1 - pred_nt[0]))
                if self.with_nms:
                    filtered_boxes = copy.deepcopy(pred_box)
                    filtered_scores = copy.deepcopy(score)
                    filtered_indices = nms(filtered_boxes, filtered_scores, 0.7)
                    filtered_boxes = filtered_boxes[filtered_indices]
                    filtered_scores = filtered_scores[filtered_indices]
                    nms_indices.append(filtered_indices)

                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    pred_box /= pred_box.new_tensor(scale_factors)
                if self.with_nms:
                    cur_predict_dict = {
                        "boxes": pred_box,
                        "scores": score,
                        "filtered_boxes": filtered_boxes,
                        "filtered_scores": filtered_scores,
                        "pred_nt": pred_nt,
                    }
                else:
                    cur_predict_dict = {"boxes": pred_box, "scores": score, "pred_nt": pred_nt}
                parts_list["pred_scores"].append(score.cpu().detach().numpy())
                parts_list["pred_boxes"].append(pred_box.cpu().detach().numpy())
                parts_list["nt"].append(pred_nt.cpu().detach().numpy())
                scores.append(score)
                pred_bboxes.append(cur_predict_dict)

        if with_mask:
            pred_masks_binary = pred_mask_.sigmoid()
            pred_masks_binary[pred_masks_binary < self.threshold] = 0.0
            pred_masks_binary[pred_masks_binary >= self.threshold] = 1.0
            for ind, (img_meta, pred_mask) in enumerate(zip(img_metas, pred_masks_binary)):
                pred_nt = 1 - nt_labels[ind]
                h_pad, w_pad = img_meta["pad_shape"][:2]
                cur_scores = scores[ind]
                valid_mask = pred_mask[cur_scores > self.score_threshold]
                mask_ = torch.any(valid_mask, dim=0).int()
                parts_list["pred_masks"].append(copy.deepcopy(pred_mask.int().cpu().detach().numpy()))
                parts_list["pred_global_mask"].append(mask_.cpu().detach().numpy())
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask_.cpu().numpy().astype(np.uint8)))

                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict

                gt_nt = img_meta["empty"]
                pred_masks.append({"pred_masks": pred_rle, "pred_nt": pred_nt, "gt_nt": gt_nt})

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks, parts_list=parts_list)

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        scores = F.softmax(box_cls, dim=-1)[:, :, 0]

        for i, (
            scores_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            results.append(result)
        return results

    def visualiation_parts(self, pred_dict, img_metas, targets, save_target_dir, extra_dict, text_attention_mask):
        index, min = 0, 50
        for i in range(text_attention_mask.size(0)):
            value = torch.sum(text_attention_mask[i])
            if value < min:
                min = value
                index = i

        save_filename = os.path.join(save_target_dir, str(self.iter))
        pred_boxes = pred_dict["pred_boxes"][index]
        pred_scores = pred_dict["pred_scores"][index]
        pred_nt = pred_dict["nt"][index]
        gt_mask = maskUtils.decode(targets["mask"][index])
        pred_global_mask = pred_dict["pred_global_mask"][index]
        pred_masks = pred_dict["pred_masks"][index]
        gt_all_box = targets["bbox"][index]
        refer_index = img_metas[index]["refer_target_index"]
        gt_box = gt_all_box[refer_index]

        # sort pred_score
        sort_index = np.argsort(pred_scores)[::-1]
        pred_scores = pred_scores[sort_index]
        pred_boxes = pred_boxes[sort_index]
        pred_masks = pred_masks[sort_index]

        if "img_selected_points" in extra_dict:
            img_selected_points = extra_dict["img_selected_points"][0]
        if "attn_map_query" in extra_dict:
            attn_map = extra_dict["attn_map_query"][index]

        expression = img_metas[index]["expression"]
        file_name = img_metas[index]["filename"]

        row_columns = self.visualize_params["row_columns"]
        # 创建一个新图像
        fig, axs = plt.subplots(
            row_columns[0],
            row_columns[1],
            figsize=(row_columns[1] * 3, row_columns[0] * 3),
        )

        H, W = gt_mask.shape
        img = Image.open(file_name)
        # img = img.convert('L')
        img = img.convert("RGB")
        img = img.resize((W, H))
        for i in range(row_columns[0]):
            for j in range(row_columns[1]):
                # axs[i, j].imshow(img, cmap="gray")
                axs[i, j].imshow(img)
                axs[i, j].axis("off")

        # 对每一行进行水平拼接
        for i in range(row_columns[0]):
            for j in range(row_columns[1]):
                idx = i * row_columns[1] + j

                if self.vis_box:
                    box = pred_boxes[idx]
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    axs[i, j].add_patch(rect)

                if self.vis_mask:
                    mask = pred_masks[idx]
                    mask_alpha = 0.5  # 调整透明度
                    axs[i, j].imshow(
                        mask,
                        cmap=ListedColormap(["none", "red"]),  # 使用红色显示 mask
                        alpha=mask_alpha,
                        interpolation="none",
                        extent=(0, mask.shape[1], mask.shape[0], 0),  # 保持 mask 对齐图像
                    )

                if pred_scores[idx] > 0.0:
                    color = "red"
                    score_text = (
                        f"{pred_scores[idx]:.2f}"
                        + "  n:"
                        + f"{float(1- pred_nt):.2f}"
                        + "  r:"
                        + f"{(pred_scores[idx]/float((1-pred_nt))):.2f}"
                    )
                    axs[i, j].text(
                        10,
                        -10,
                        score_text,
                        color=color,
                        fontsize=10,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                    )

                if "attn_map_query" in extra_dict:
                    tokenized_words = img_metas[index]["tokenized_words"]
                    text_attn_map = attn_map[idx]
                    assert len(tokenized_words) == (50 - min) - 2
                    if len(tokenized_words) < len(text_attn_map) - 2:
                        targeted_text_attn_map = text_attn_map[1 : len(tokenized_words) + 1]
                    else:
                        targeted_text_attn_map = text_attn_map[1:-1]
                    targeted_word = tokenized_words[torch.argmax(targeted_text_attn_map)]
                    axs[i, j].text(
                        0.5,
                        -0.1,
                        targeted_word,  # 显示在图片上方
                        color="blue",
                        fontsize=12,
                        fontweight="bold",
                        verticalalignment="top",
                        horizontalalignment="center",
                        transform=axs[i, j].transAxes,  # 使用 Axes 的坐标系
                    )

        save_filename_query = save_filename + "-{}-querymap.jpg".format(expression)
        plt.savefig(save_filename_query)

        box_gt = (gt_box.cpu().detach().numpy()).astype(np.int32)
        mask_gt = gt_mask.astype(np.int32)
        mask_gt = Image.fromarray(mask_gt * 255)
        image_gt = Image.new("RGB", (W, H))
        image_gt.paste(mask_gt)
        draw_gt = ImageDraw.Draw(image_gt)
        for box in box_gt:
            draw_gt.rectangle(list(box), outline="red", width=2)

        mask_pred = pred_global_mask.astype(np.int32)
        mask_pred = Image.fromarray(mask_pred * 255)
        image_pred = Image.new("RGB", (W, H))
        image_pred.paste(mask_pred)
        draw_pred = ImageDraw.Draw(image_pred)
        if self.vis_box:
            filterd_pred_box = pred_boxes[np.where(pred_scores > self.score_threshold)]
            box_pred = filterd_pred_box.astype(np.int32)
            for box in box_pred:
                draw_pred.rectangle(list(box), outline="blue", width=2)

        imshow_image_nums = 4

        if "attn_map_query" in extra_dict:
            attn_map = attn_map[:, 1 : (50 - min) - 1]
            attn_map = attn_map.cpu().detach().numpy()
            attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
            attn_map = (attn_map * 255).astype(np.uint8)
            attn_map = cv2.resize(attn_map, (W, H))
            attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
            attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)
            attn_map = Image.fromarray(attn_map)
            image_attn = Image.new("RGB", (W, H))
            image_attn.paste(attn_map)
            imshow_image_nums += 1

        imge_allgt = Image.open(file_name)
        imge_allgt = imge_allgt.resize((W, H))
        draw_alldet = ImageDraw.Draw(imge_allgt)
        for ind, box in enumerate(gt_all_box):
            color = colors[ind]
            draw_alldet.rectangle(list(box), outline=color, width=2)

        img_source = Image.open(file_name)
        img_source = img_source.resize((W, H))
        concat_image = Image.new("RGB", (W * imshow_image_nums + (imshow_image_nums - 1) * 10, H), "white")
        concat_image.paste(img_source, (0, 0))
        concat_image.paste(image_gt, (W + 10, 0))
        concat_image.paste(image_pred, (2 * W + 20, 0))
        concat_image.paste(imge_allgt, (3 * W + 30, 0))
        if "attn_map_query" in extra_dict:
            concat_image.paste(image_attn, (5 * W + 50, 0))
        save_filename_src = save_filename + "-{}-src.jpg".format(expression)
        concat_image.save(save_filename_src)
