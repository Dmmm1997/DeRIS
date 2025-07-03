import argparse
import torch.distributed as dist

from deris.apis import evaluate_model, set_random_seed
from deris.datasets import build_dataset, build_dataloader
from deris.models import build_model, ExponentialMovingAverage
from deris.utils import (
    get_root_logger,
    load_checkpoint,
    init_dist,
    is_main,
    load_pretrained_checkpoint,
)

from mmcv.runner import get_dist_info
from mmcv.utils import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel
import os
import pandas as pd
import gc

try:
    import apex
except:
    pass


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()
    work_dir = os.path.dirname(cfg.load_from)
    if is_main():
        logger = get_root_logger(log_file=os.path.join(work_dir, "test_log.txt"))
        # logger = get_root_logger()
        logger.info(cfg.pretty_text)

    if cfg.dataset == "RRefCOCO":
        prefix = ["val_rrefcoco", "val_rrefcoco+", "val_rrefcocog"]
        datasets_cfgs = [
            # cfg.data.train,
            cfg.data.val_rrefcoco,
            cfg.data.val_rrefcocoplus,
            cfg.data.val_rrefcocog,
        ]
    elif cfg.dataset == "MixedSeg":
        prefix = [
            "val_refcoco_unc",
            "testA_refcoco_unc",
            "testB_refcoco_unc",
            "val_refcocoplus_unc",
            "testA_refcocoplus_unc",
            "testB_refcocoplus_unc",
            "val_refcocog_umd",
            "test_refcocog_umd",
        ]
        datasets_cfgs = [
            cfg.data.val_refcoco_unc,
            cfg.data.testA_refcoco_unc,
            cfg.data.testB_refcoco_unc,
            cfg.data.val_refcocoplus_unc,
            cfg.data.testA_refcocoplus_unc,
            cfg.data.testB_refcocoplus_unc,
            cfg.data.val_refcocog_umd,
            cfg.data.test_refcocog_umd,
        ]
    else:
        prefix = ["val"]
        datasets_cfgs = [cfg.data.val]
        if hasattr(cfg.data, "testA") and hasattr(cfg.data, "testB"):
            datasets_cfgs.append(cfg.data.testA)
            datasets_cfgs.append(cfg.data.testB)
            prefix.extend(["testA", "testB"])
        elif hasattr(cfg.data, "test"):
            datasets_cfgs.append(cfg.data.test)
            prefix.extend(["test"])
    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))
    cfg.model.mask_save_target_dir = work_dir
    # cfg.model.threshold = cfg.threshold
    model = build_model(cfg.model)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank])
    if cfg.load_from:
        load_checkpoint(model, load_from=cfg.load_from)
    elif cfg.finetune_from:
        # hacky way
        load_pretrained_checkpoint(model, cfg.finetune_from, amp=cfg.use_fp16)
    if cfg.dataset == "GRefCOCO":
        excel_results = {
            "F1score": [],
            "Nacc": [],
            "Tacc": [],
            "gIoU": [],
            "cIoU": [],
            "PR@7-9": [],
        }
    elif cfg.dataset == "RefZOM":
        excel_results = {
            "mIoU": [],
            "oIoU": [],
            "macc": [],
            "PR@5-9": [],
        }
    elif cfg.dataset == "RRefCOCO":
        excel_results = {
            "mIoU": [],
            "oIoU": [],
            "mRR": [],
            "rIoU": [],
        }
    elif cfg.dataset == "MixedSeg":
        excel_results = {
            "mIoU": [],
            "oIoU": [],
            "DetACC": [],
            "MaskACC@0.5-0.9": [],
            "DetACC@0.5-0.9": [],
        }
    index_names = []
    for eval_loader, _prefix in zip(dataloaders, prefix):
        if is_main():
            logger = get_root_logger()
            logger.info(f"DeRIS - evaluating set {_prefix}")

        if cfg.dataset == "GRefCOCO":
            set_F1score, set_N_acc, set_T_acc, set_gIoU, set_cIoU, set_PR_list_res = evaluate_model(
                _prefix, cfg, model, eval_loader
            )
            if is_main():
                excel_results["F1score"].append("{:.2f}".format(set_F1score))
                excel_results["Nacc"].append("{:.2f}".format(set_N_acc))
                excel_results["Tacc"].append("{:.2f}".format(set_T_acc))
                excel_results["gIoU"].append("{:.2f}".format(set_gIoU))
                excel_results["cIoU"].append("{:.2f}".format(set_cIoU))
                excel_results["PR@7-9"].append("{:.2f},{:.2f},{:.2f}".format(*set_PR_list_res))
                index_names.append(_prefix)
        elif cfg.dataset == "RefZOM":
            set_mIoU, set_oIoU, set_macc, set_PR_list_res = evaluate_model(_prefix, cfg, model, eval_loader)
            if is_main():
                excel_results["mIoU"].append("{:.2f}".format(set_mIoU))
                excel_results["oIoU"].append("{:.2f}".format(set_oIoU))
                excel_results["macc"].append("{:.2f}".format(set_macc))
                excel_results["PR@5-9"].append("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*set_PR_list_res))
                index_names.append(_prefix)
        elif cfg.dataset == "RRefCOCO":
            set_mIoU, set_oIoU, set_mRR, set_rIoU = evaluate_model(_prefix, cfg, model, eval_loader)
            if is_main():
                excel_results["mIoU"].append("{:.2f}".format(set_mIoU))
                excel_results["oIoU"].append("{:.2f}".format(set_oIoU))
                excel_results["mRR"].append("{:.2f}".format(set_mRR))
                excel_results["rIoU"].append("{:.2f}".format(set_rIoU))
                index_names.append(_prefix)
        elif cfg.dataset == "MixedSeg":
            det_acc, mask_miou, mask_oiou, mask_acc, det_accs = evaluate_model(_prefix, cfg, model, eval_loader)
            if is_main():
                excel_results["DetACC"].append("{:.2f}".format(det_acc))
                excel_results["mIoU"].append("{:.2f}".format(mask_miou))
                excel_results["oIoU"].append("{:.2f}".format(mask_oiou))
                excel_results["MaskACC@0.5-0.9"].append("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*mask_acc))
                excel_results["DetACC@0.5-0.9"].append("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*det_accs))
                index_names.append(_prefix)
        del eval_loader
        gc.collect()

    if is_main():
        df = pd.DataFrame(excel_results, index=index_names)
        target_excel_path = os.path.join(
            work_dir,
            "refer_output_thr{}_{}_{}_{}.xlsx".format(
                cfg.model.post_params["score_threshold"],
                "nms" if cfg.model.post_params["with_nms"] else "no-nms",
                "sw" if cfg.model.post_params["score_weighted"] else "no-sw",
                cfg.model.post_params["mask_threshold"],
            ),
        )
        df.to_excel(target_excel_path, engine="openpyxl")
        logger.info("sucessfully save the results to {} !!!".format(target_excel_path))

    if cfg.distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-test")
    parser.add_argument("config", help="test configuration file path.")
    parser.add_argument("--load-from", help="load from the saved .pth checkpoint, only used in validation.")
    parser.add_argument("--finetune-from", help="load from the pretrained checkpoint, only used in validation.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--mask-threshold", default=0.5, type=float, help="mask positive and negative threshold")
    parser.add_argument("--score-threshold", default=0.7, type=float, help="query score threshold")
    parser.add_argument("--nms", default=False, type=bool, help="nms")
    parser.add_argument(
        "--score-weighted",
        default=False,
        type=bool,
        help="pred-nt score multiply the instance score",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.launcher = args.launcher

    cfg.model.visual_mode = "test"

    cfg.model.post_params["mask_threshold"] = args.mask_threshold
    cfg.model.post_params["score_threshold"] = args.score_threshold
    cfg.model.post_params["with_nms"] = args.nms
    cfg.model.post_params["score_weighted"] = args.score_weighted

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == "__main__":
    main()
