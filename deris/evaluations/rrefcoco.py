import numpy as np
import torch
import torch.nn.functional as F
import time
import pycocotools.mask as maskUtils
import json


class AverageMeter:
    """
    Compute and stores the average and current value
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rrefcoco_evaluate(
    pred_bboxes, pred_masks, gt_masks, threshold=0.7, cur_dataset="val_rrefcoco", instance_index_file="data/seqtr_type/annotations/rrefcoco/instance_index.json"
):
    """
    evaluate R-RIS with new metrics.
    """
    with open(instance_index_file, "r") as f:
        instance_index = json.load(f)[cur_dataset]

    def computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U

    seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(seg_iou_list), dtype=np.int32)
    I_meter = AverageMeter()
    U_meter = AverageMeter()
    mIOU_meter = AverageMeter()
    r_meter = AverageMeter()
    rIoU_meter = AverageMeter()
    seg_total = 0
    assert instance_index[-1][-1] == len(pred_bboxes)
    for start_index, end_index in instance_index:
        start_index, end_index = int(start_index), int(end_index)
        pred_bboxes_instance = pred_bboxes[start_index:end_index]
        pred_masks_instance = pred_masks[start_index:end_index]

        I, U, IoU = 0, 0, 0
        cur_total = 0
        for ind, (pred_bbox, eval_sample) in enumerate(zip(pred_bboxes_instance, pred_masks_instance)):
            pred_mask = maskUtils.decode(eval_sample["pred_masks"])[None]
            gt_mask = maskUtils.decode(gt_masks[ind+start_index])[None]
            ref_result = {}
            ref_result["gt_nt"] = eval_sample["gt_nt"]
            ref_result["pred_nt"] = ((pred_bbox["scores"] > threshold).sum() == 0).item()

            if not ref_result["gt_nt"]:  # Targeted Samples
                temp_I, temp_U = computeIoU(pred_mask, gt_mask)
                I = I + temp_I
                U = U + temp_U
                IoU = IoU + temp_I * 1.0 / temp_U
                for n_eval_iou in range(len(seg_iou_list)):
                    eval_seg_iou = seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += IoU >= eval_seg_iou
                seg_total += 1
                cur_total += 1

        IoU = IoU / cur_total
        I_meter.update(I)
        U_meter.update(U)
        mIOU_meter.update(IoU)

        TN, FP = 0, 0
        for ind, (pred_bbox, eval_sample) in enumerate(zip(pred_bboxes_instance, pred_masks_instance)):
            pred_mask = maskUtils.decode(eval_sample["pred_masks"])[None]
            gt_mask = maskUtils.decode(gt_masks[ind])[None]
            ref_result = {}
            ref_result["gt_nt"] = eval_sample["gt_nt"]
            ref_result["pred_nt"] = ((pred_bbox["scores"] > threshold).sum() == 0).item()

            if ref_result["gt_nt"]:  # No-target Samples
                if not ref_result["pred_nt"]:
                    FP += 1
                    U = U + pred_mask.sum()
                else:
                    TN += 1
        # if TN + FP == 0:
        #     r = 1
        # else:
        r = TN / (TN + FP)
        rIoU = I / U
        r_meter.update(r)
        rIoU_meter.update(rIoU)

    mIoU = 100 * mIOU_meter.avg
    oIoU = 100 * float(I_meter.sum) / float(U_meter.sum)
    r = 100 * r_meter.avg
    rIoU = 100 * rIoU_meter.avg
    PR_list = []
    for n_eval_iou in range(len(seg_iou_list)):
        PR_list.append(seg_correct[n_eval_iou] * 100.0 / seg_total)
    return mIoU, oIoU, r, rIoU
