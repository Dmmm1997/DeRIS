import itertools
import time
import torch
import numpy

import pycocotools.mask as maskUtils
from deris.datasets import extract_data
from deris.evaluations.grefcoco import grec_evaluate_f1_nacc, gres_evaluate
from deris.evaluations.refzom import refzom_evaluate
from deris.evaluations.rrefcoco import rrefcoco_evaluate
from deris.models.postprocess.nms import nms
from deris.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict
from mmdet.core import BitmapMasks
import numpy as np
import copy
from deris.utils.dist import all_gather, synchronize
from tqdm import tqdm


def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
    gt_mask (list[RLE]):
    pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, is_crowd)
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou)

    return mask_iou


def mask_overlaps_withIU_RLE(gt_masks, pred_masks, is_crowds):
    # decode the mask
    pred_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in pred_masks], dim=0)
    gt_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in gt_masks], dim=0)
    # pred_mask = pred_mask.argmax(1)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=-1)
    union = torch.stack(
        [
            pred_mask_.sum() if is_crowd else (pred_mask_ + gt_mask_).sum() - inters
            for pred_mask_, gt_mask_, is_crowd, inters in zip(pred_mask, gt_mask, is_crowds, intersection)
        ],
        dim=0,
    )
    intersection = intersection.cuda()
    union = union.cuda()

    # union = torch.sum(torch.add(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=1).cuda() - intersection
    iou = torch.tensor([i / u if u >= 1 else 0 for i, u in zip(intersection, union)]).cuda()
    return iou, intersection, union


def mask_overlaps_withIU(gt_masks, pred_masks, is_crowd):
    # decode the mask
    pred_mask = torch.from_numpy(maskUtils.decode(pred_masks)[None])
    gt_mask = torch.from_numpy(maskUtils.decode(gt_masks)[None])
    # pred_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in pred_masks], dim=0)
    # gt_mask = torch.concat([torch.from_numpy(maskUtils.decode(gt_rle)[None]) for gt_rle in gt_masks], dim=0)
    # pred_mask = pred_mask.argmax(1)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=-1).cuda()
    union = torch.sum(torch.add(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=1).cuda() - intersection
    iou = torch.tensor([i / u if u >= 1 else 0 for i, u in zip(intersection, union)]).cuda()
    return iou, intersection, union


def accuracy(pred_bboxes, gt_bboxes, pred_masks, gt_masks, is_crowd=None, device="cuda:0"):
    # eval_det = pred_bboxes is not None

    if pred_bboxes == []:
        eval_det = False
    else:
        eval_det = True
    eval_mask = pred_masks is not None

    det_acc_list, mask_iou_list, mask_acc_list, mask_I_list, mask_U_list, det_acc_at_thrs_list = [], [], [], [], [], []

    if eval_det:
        for ind, (ppred_bbox, gt_bbox) in enumerate(zip(pred_bboxes, gt_bboxes)):
            if "filtered_scores" in ppred_bbox:
                sorted_scores_boxes = sorted(
                    zip(ppred_bbox["filtered_scores"].tolist(), ppred_bbox["filtered_boxes"].tolist()), reverse=True
                )
            else:
                sorted_scores_boxes = sorted(
                    zip(ppred_bbox["scores"].tolist(), ppred_bbox["boxes"].tolist()), reverse=True
                )
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            sorted_scores = torch.cat([torch.as_tensor(x).view(1, 1) for x in sorted_scores])
            max_index = torch.argmax(sorted_scores)
            pred_bbox = sorted_boxes[max_index].view(1, 4)
            det_acc_at_thrs = torch.full((5,), -1.0, device=device)
            det_acc = torch.tensor([0.0], device=device)
            bbox_iou = torch.tensor([0.0], device=device)
            bbox_iou = bbox_overlaps(gt_bbox.view(-1, 4).unsqueeze(0), pred_bbox.unsqueeze(0), is_aligned=True)
            for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
                det_acc_at_thrs[i] = (bbox_iou >= iou_thr).float().mean() * 100.0
            det_acc = (bbox_iou >= 0.5).float().mean()
            det_acc_list.append(det_acc)
            det_acc_at_thrs_list.append(det_acc_at_thrs)

    if eval_mask:
        for ind, eval_sample in enumerate(pred_masks):
            mask_iou = torch.tensor([0.0], device=device)
            mask_acc_at_thrs = torch.full((5,), -1.0, device=device)
            I, U = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            # mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
            mask_iou, I, U = mask_overlaps_withIU(gt_masks[ind], eval_sample["pred_masks"], is_crowd)
            for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
                mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean() * 100.0
            mask_iou_list.append(mask_iou)
            mask_acc_list.append(mask_acc_at_thrs)
            mask_I_list.append(I.float())
            mask_U_list.append(U.float())
    if eval_det:
        det_acc = sum(det_acc_list) / len(det_acc_list)
        det_accs = torch.vstack(det_acc_at_thrs_list).mean(dim=0).tolist()
    else:
        det_acc = torch.tensor(0.0, device=device)
        det_accs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    if eval_mask:
        mask_miou = torch.cat(mask_iou_list).mean().item()
        mask_I = torch.cat(mask_I_list).mean().item()
        mask_U = torch.cat(mask_U_list).mean().item()
        mask_oiou = 100.0 * mask_I / mask_U
        mask_acc = torch.vstack(mask_acc_list).mean(dim=0).tolist()
    else:
        mask_miou = torch.tensor(0.0, device=device)
        mask_oiou = torch.tensor(0.0, device=device)
        mask_acc = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)

    return det_acc * 100.0, mask_miou * 100.0, mask_oiou, mask_acc, det_accs


def evaluate_model(prefex, cfg, model, loader):
    model.eval()
    device = list(model.parameters())[0].device

    batches = len(loader)

    with_bbox, with_mask = False, False
    pred_bboxes, gt_bboxes, pred_masks, gt_masks, is_crowds, img_meta_total, cover_acc = [], [], [], [], [], [], []
    start = time.time()
    pbar = tqdm(total=batches)
    with torch.no_grad():
        for batch, inputs in enumerate(loader):
            # if batch>100:
            #     break
            gt_bbox, gt_mask = [], []
            if "gt_bbox" in inputs:
                if isinstance(inputs["gt_bbox"], torch.Tensor):
                    inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                    gt_bbox = copy.deepcopy(inputs["gt_bbox"])
                else:
                    gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
                with_bbox = True

            if "gt_mask_rle" in inputs:
                gt_mask = inputs.pop("gt_mask_rle").data[0]
                with_mask = True

            img_metas = inputs["img_metas"].data[0]
            refer_box = [gt_bbox_[img_meta["refer_target_index"]] for img_meta, gt_bbox_ in zip(img_metas, gt_bbox)]

            if not cfg.distributed:
                inputs = extract_data(inputs)

            predictions = model(
                **inputs,
                return_loss=False,
                gt_mask=gt_mask,
                rescale=False,
                with_bbox=with_bbox,
                with_mask=with_mask,
            )
            if cfg.distributed:
                synchronize()
                pred_bboxes.extend(list(itertools.chain(*all_gather(predictions.pop("pred_bboxes")))))
                pred_masks.extend(list(itertools.chain(*all_gather(predictions.pop("pred_masks")))))
                gt_bboxes.extend(list(itertools.chain(*all_gather(refer_box))))
                gt_masks.extend(list(itertools.chain(*all_gather(gt_mask))))
                img_meta_total.extend(list(itertools.chain(*all_gather(img_metas))))
                # cover_acc.extend(list(itertools.chain(*all_gather(predictions.pop("bs_acc_list")))))
            else:
                pred_bboxes.extend(predictions.pop("pred_bboxes"))
                pred_masks.extend(predictions.pop("pred_masks"))
                gt_bboxes.extend(refer_box)
                gt_masks.extend(gt_mask)
                img_meta_total.extend(img_metas)
                # cover_acc.extend(predictions.pop("bs_acc_list"))
            if is_main():
                pbar.update()
    # cover_ACC = sum(cover_acc) / len(cover_acc) *100.0
    if cfg.dataset == "GRefCOCO":
        F1_score_rec, N_acc, T_acc = grec_evaluate_f1_nacc(
            pred_bboxes,
            gt_bboxes,
            img_meta_total,
            thresh_score=cfg.model.post_params["score_threshold"],
            thresh_iou=0.5,
            thresh_F1=1.0,
        )
        gIoU_res, cIoU_res, T_acc_res, N_acc_res, PR_list_res = gres_evaluate(
            pred_bboxes, pred_masks, gt_masks, threshold=cfg.model.post_params["score_threshold"]
        )
        if is_main():
            logger = get_root_logger()
            logger.info(
                f"------------ validate ------------  "
                + f"time: {(time.time() - start):.2f}, "
                + f"F1score: {F1_score_rec:.2f}, "
                + f"Nacc: {N_acc:.2f}, "
                + f"Tacc: {T_acc:.2f}, "
                + f"gIoU: {gIoU_res:.2f}, "
                + f"cIoU: {cIoU_res:.2f}, "
                # + f"m_Tacc: {T_acc_res:.2f} "
                # + f"cover_ACC: {cover_ACC:.2f}, "
                + f"MaskACC@0.7-0.9: [{PR_list_res[0]:.2f}, {PR_list_res[1]:.2f}, {PR_list_res[2]:.2f}"
            )
        return F1_score_rec, N_acc, T_acc, gIoU_res, cIoU_res, PR_list_res
    elif cfg.dataset == "RefZOM":
        mIoU, oIoU, macc, PR_list = refzom_evaluate(
            pred_bboxes, pred_masks, gt_masks, threshold=cfg.model.post_params["score_threshold"]
        )
        if is_main():
            logger = get_root_logger()
            logger.info(
                f"------------ validate ------------"
                + f"time: {(time.time() - start):.2f}, "
                + f"mIoU: {mIoU:.2f}, "
                + f"oIoU: {oIoU:.2f}, "
                + f"macc: {macc:.2f}, "
                # + f"cover_ACC: {cover_ACC:.2f}, "
                + f"MaskACC@0.5-0.9: [{PR_list[0]:.2f}, {PR_list[1]:.2f}, {PR_list[2]:.2f}, {PR_list[3]:.2f}, {PR_list[4]:.2f}"
            )
        return mIoU, oIoU, macc, PR_list
    elif cfg.dataset == "RRefCOCO":
        mIoU, oIoU, mRR, rIoU = rrefcoco_evaluate(
            pred_bboxes,
            pred_masks,
            gt_masks,
            threshold=cfg.model.post_params["score_threshold"],
            cur_dataset=prefex,
            instance_index_file="data/seqtr_type/annotations/rrefcoco/instance_index.json",
        )
        if is_main():
            logger = get_root_logger()
            logger.info(
                f"------------ validate ------------"
                + f"time: {(time.time() - start):.2f}, "
                + f"mIoU: {mIoU:.2f}, "
                + f"oIoU: {oIoU:.2f}, "
                + f"mRR: {mRR:.2f}, "
                # + f"cover_ACC: {cover_ACC:.2f}, "
                + f"rIoU: {rIoU:.2f}"
            )
        return mIoU, oIoU, mRR, rIoU
    # '''
    elif cfg.dataset == "MixedSeg":
        det_acc, mask_miou, mask_oiou, mask_acc, det_accs = accuracy(
            pred_bboxes, gt_bboxes, pred_masks, gt_masks, is_crowd=None, device=device
        )

        if is_main():
            logger = get_root_logger()
            logger.info(
                f"------------ validate ------------  "
                + f"time: {(time.time() - start):.2f}, "
                + f"DetACC: {det_acc:.2f}, "
                + f"mIoU: {mask_miou:.2f}, "
                + f"oIoU: {mask_oiou:.2f}, "
                + f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}]"
                + f"DetACC@0.5-0.9: [{det_accs[0]:.2f}, {det_accs[1]:.2f}, {det_accs[2]:.2f},  {det_accs[3]:.2f},  {det_accs[4]:.2f}]"
            )
        return det_acc, mask_miou, mask_oiou, mask_acc, det_accs


# '''
