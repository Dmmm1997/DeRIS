import time
import argparse
import os.path as osp
import torch.distributed as dist

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel

from deris.core import build_optimizer, build_scheduler
from deris.datasets import build_dataset, build_dataloader
from deris.models import build_model, ExponentialMovingAverage
from deris.apis import set_random_seed, train_model, evaluate_model
from deris.utils import (
    get_root_logger,
    load_checkpoint,
    save_checkpoint,
    load_pretrained_checkpoint,
    is_main,
    init_dist,
)
import wandb
import copy
import warnings
import torch

# from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")

try:
    import apex
except:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-train")
    parser.add_argument("config", help="training configuration file path.")
    parser.add_argument("--debug", default=False, help="training configuration file path.")
    parser.add_argument("--work-dir", help="directory of config file, training logs, and checkpoints.")
    parser.add_argument("--resume-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument("--load-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument(
        "--finetune-from",
        help="finetune from the saved .pth checkpoint, only used after SeqTR has been pre-trained on the merged dadtaset.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
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


def find_linear_layers(model, lora_target_modules=["q_proj", "v_proj"], target_module_list=[]):
    cur_target_module_list = copy.deepcopy(target_module_list)
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and any([x in name for x in cur_target_module_list])
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    # if cfg.distributed and cfg.dataset == "RRefCOCO":
    #     raise NotImplementedError("Distributed training is not supported for RRefCOCO dataset.")
    cfg.rank, cfg.world_size = get_dist_info()
    if is_main():
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        wandb.init(project="deris-seg-offline", tags=[cfg.tag_name], name=cfg.wandb_name, mode="offline")
        logger = get_root_logger(log_file=osp.join(cfg.work_dir, str(cfg.timestamp) + "_train_log.txt"))
        logger.info(cfg.pretty_text)
        cfg.dump(osp.join(cfg.work_dir, f"{cfg.timestamp}_" + osp.basename(cfg.config)))

    datasets_cfgs = [cfg.data.train]
    if cfg.dataset == "Mixed":
        items = [
            "val_refcoco_unc",
            "val_refcocoplus_unc",
            "val_refcocog_umd",
            "val_referitgame_berkeley",
            "val_flickr30k",
        ]
        for item in items:
            if getattr(cfg.data, item, None):
                datasets_cfgs += [getattr(cfg.data, item)]
    elif cfg.dataset == "MixedSeg":
        items = ["val_refcoco_unc", "val_refcocoplus_unc", "val_refcocog_umd", "val_refcocog_google"]
        for item in items:
            if getattr(cfg.data, item, None):
                datasets_cfgs += [getattr(cfg.data, item)]
    elif cfg.dataset == "RRefCOCO":
        items = ["val_rrefcoco", "val_rrefcoco+", "val_rrefcocog"]
        datasets_cfgs += [cfg.data.val_rrefcoco, cfg.data.val_rrefcocoplus, cfg.data.val_rrefcocog]
    else:
        items = ["val"]
        datasets_cfgs += [cfg.data.val]

    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    cfg.model.mask_save_target_dir = cfg.work_dir
    # cfg.model.threshold = cfg.threshold
    # cfg.model.box_threshold = cfg.score_threshold
    model = build_model(cfg.model)
    model = model.cuda()
    # if getattr(cfg, "lora_config", None) is not None and cfg.lora_config["enable"]:
    #     lora_r = cfg.lora_config["r"]
    #     lora_alpha = cfg.lora_config["lora_alpha"]
    #     lora_dropout = cfg.lora_config["lora_dropout"]
    #     target_module_list = cfg.lora_config.get("target_modules", [])
    #     lora_target_modules = cfg.lora_config.get("lora_target_modules", ["q_proj", "v_proj"])

    #     lora_modules = find_linear_layers(
    #         model, lora_target_modules=lora_target_modules, target_module_list=target_module_list
    #     )

    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=lora_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="SEQ_CLS",
    #     )
    #     model = get_peft_model(model, lora_config)

    #     model.print_trainable_parameters()

    #     for n, p in model.named_parameters():
    #         if all([x not in n for x in ["beit3", "hier_vision_encoder"]]):
    #             p.requires_grad = True

    train_params = [
        {
            "params": [p for n, p in model.named_parameters() if "vis_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_vis_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "lan_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_lan_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "hier_vision_encoder" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_hier_vis_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "pixel_decoder" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_pixel_decoder"),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "hier_vision_encoder" not in n
                and "vis_enc" not in n
                and "lan_enc" not in n
                and "pixel_decoder" not in n  # and "lan_enc" not in n
                and p.requires_grad
            ],
            "lr": cfg.optimizer_config.lr,
        },
    ]

    optimizer = build_optimizer(cfg.optimizer_config, train_params)
    scheduler = build_scheduler(cfg.scheduler_config, optimizer)

    if is_main():
        print("Trainable parameters with gradients:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Name: {name}, Shape: {param.shape}")

    if cfg.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True

    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank], find_unused_parameters=True)
    (
        start_epoch,
        best_F1score,
        best_Nacc_rec,
        best_gIoU,
        best_cIoU,
        best_mRR,
        best_rIoU,
        best_mIoU,
        best_oIoU,
        best_macc,
        best_d_acc,
    ) = (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if cfg.resume_from:
        start_epoch, _, _, flag = load_checkpoint(
            model, cfg.resume_from, amp=cfg.use_fp16, optimizer=optimizer, scheduler=scheduler
        )
    elif cfg.finetune_from:
        load_pretrained_checkpoint(model, cfg.finetune_from, amp=cfg.use_fp16, dataset=cfg.dataset)
    elif cfg.load_from:
        start_epoch, best_F1score, best_gIoU, flag = load_checkpoint(
            model, load_from=cfg.load_from, dataset=cfg.dataset
        )

    import time

    begin_time = time.time()
    for epoch in range(start_epoch + 1, cfg.scheduler_config.max_epoch):
        start_time = time.time()
        train_model(epoch, cfg, model, optimizer, dataloaders[0])
        this_epoch_train_time = int(time.time() - start_time)
        if is_main():
            logger.info("this_epoch_train_time={}m-{}s".format(this_epoch_train_time // 60, this_epoch_train_time % 60))

        if epoch % cfg.evaluate_interval == 0 and epoch >= cfg.start_evaluate_epoch:
            F1score, Nacc_rec, gIoU, cIoU = 0, 0, 0, 0
            mIoU, oIoU, macc = 0, 0, 0
            mRR, rIoU = 0, 0
            d_acc = 0

            for _prefix, _loader in zip(items, dataloaders[1:]):
                if is_main():
                    logger.info("Evaluating dataset: {}".format(_loader.dataset.which_set))
                if cfg.dataset == "GRefCOCO":
                    set_F1score, set_Nacc_rec, set_T_acc_rec, set_gIoU, set_cIoU, set_PR_list_res = evaluate_model(
                        _prefix, cfg, model, _loader
                    )
                    F1score += set_F1score
                    Nacc_rec += set_Nacc_rec
                    gIoU += set_gIoU
                    cIoU += set_cIoU
                elif cfg.dataset == "RefZOM":
                    set_mIoU, set_oIoU, set_macc, set_PR_list_res = evaluate_model(_prefix, cfg, model, _loader)
                    mIoU += set_mIoU
                    oIoU += set_oIoU
                    macc += set_macc
                elif cfg.dataset == "RRefCOCO":
                    # set_mIoU, set_oIoU, set_mRR, set_rIoU = evaluate_model(_prefix, cfg, model, _loader)
                    set_mIoU, set_oIoU, set_mRR, set_rIoU = 0, 0, 0, 0
                    mIoU += set_mIoU
                    oIoU += set_oIoU
                    mRR += set_mRR
                    rIoU += set_rIoU
                elif cfg.dataset == "MixedSeg":
                    set_det_acc, set_mIoU, set_oIoU, set_macc, set_det_accs = evaluate_model(
                        _prefix, cfg, model, _loader
                    )
                    d_acc += set_det_acc
                    mIoU += set_mIoU
                    oIoU += set_oIoU

            if cfg.dataset == "GRefCOCO":
                F1score /= len(dataloaders[1:])
                Nacc_rec /= len(dataloaders[1:])
                gIoU /= len(dataloaders[1:])
                cIoU /= len(dataloaders[1:])
            elif cfg.dataset == "RefZOM":
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                macc /= len(dataloaders[1:])
            elif cfg.dataset == "RRefCOCO":
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                mRR /= len(dataloaders[1:])
                rIoU /= len(dataloaders[1:])
            elif cfg.dataset == "MixedSeg":
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                d_acc /= len(dataloaders[1:])

            if is_main():
                this_epoch_total_time = int(time.time() - start_time)
                logger.info(
                    "this_epoch_total_time={}m-{}s".format(this_epoch_total_time // 60, this_epoch_total_time % 60)
                )
                total_time = int(time.time() - begin_time)
                logger.info("total_time={}m-{}s".format(total_time // 60, total_time % 60))

            if is_main() and epoch >= cfg.start_save_checkpoint:
                if cfg.dataset == "GRefCOCO":
                    saved_info = {
                        "epoch": epoch,
                        "F1score": F1score,
                        "Nacc_rec": Nacc_rec,
                        "gIoU": gIoU,
                        "cIoU": cIoU,
                        "best_F1score": best_F1score,
                        "best_Nacc_rec": best_Nacc_rec,
                        "best_gIoU": best_gIoU,
                        "best_cIoU": best_cIoU,
                    }
                    save_compare_det = "F1score"
                    save_compare_seg = "cIoU"
                elif cfg.dataset == "RefZOM":
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "macc": macc,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                        "best_macc": best_macc,
                    }
                    save_compare_det = "mIoU"
                    save_compare_seg = "macc"
                elif cfg.dataset == "RRefCOCO":
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "mRR": mRR,
                        "rIoU": rIoU,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                        "best_mRR": best_mRR,
                        "best_rIoU": best_rIoU,
                    }
                    save_compare_det = "mIoU"
                    save_compare_seg = "mRR"
                if cfg.dataset == "MixedSeg":
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "d_acc": d_acc,
                        "best_d_acc": best_d_acc,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                    }
                    save_compare_det = "d_acc"
                    save_compare_seg = "mIoU"

                save_checkpoint(
                    cfg.work_dir,
                    cfg.save_interval,
                    model,
                    optimizer,
                    scheduler,
                    saved_info,
                    save_key_metric={"Det": save_compare_det, "Seg": save_compare_seg},
                )
            if cfg.dataset == "GRefCOCO":
                best_F1score = max(F1score, best_F1score)
                best_gIoU = max(gIoU, best_gIoU)
                best_cIoU = max(cIoU, best_cIoU)
            elif cfg.dataset == "RefZOM":
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_macc = max(macc, best_macc)
            elif cfg.dataset == "RRefCOCO":
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_mRR = max(mRR, best_mRR)
                best_rIoU = max(rIoU, best_rIoU)
            elif cfg.dataset == "MixedSeg":
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_d_acc = max(d_acc, best_d_acc)

        scheduler.step()

        if cfg.distributed:
            dist.barrier()

    if cfg.distributed:
        dist.destroy_process_group()

    wandb.finish()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # cfg.work_dir = f"./work_dir/{cfg.timestamp}_" + osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f"./work_dir/" + args.config.split("configs/")[-1].split(".py")[0]
    if is_main():
        cfg.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, f"{cfg.timestamp}")
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.finetune_from is not None:
        cfg.finetune_from = args.finetune_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    cfg.launcher = args.launcher
    cfg.config = args.config
    cfg.debug = args.debug
    cfg.wandb_name = "-".join(cfg.work_dir.split("/")[3:-1])
    cfg.tag_name = "-".join(cfg.work_dir.split("/")[3:-2])
    cfg.score_threshold = cfg.model.post_params["score_threshold"]

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == "__main__":
    main()
