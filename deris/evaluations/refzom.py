import itertools
import time
import torch
import numpy

import pycocotools.mask as maskUtils
from deris.datasets import extract_data
from deris.models.postprocess.nms import nms
from deris.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict
from mmdet.core import BitmapMasks
import numpy as np
import copy
from tqdm import tqdm


def refzom_evaluate(pred_bboxes, pred_masks, gt_masks, threshold=0.7):
    """
    prediction should have the result of pred_nt, gt_nt
    """

    def computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U

    mean_acc, mean_IoU = [], []
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    for ind, (pred_bbox, eval_sample) in enumerate(zip(pred_bboxes, pred_masks)):
        pred_mask = maskUtils.decode(eval_sample["pred_masks"])[None]
        gt_mask = maskUtils.decode(gt_masks[ind])[None]
        ref_result = {}
        ref_result["gt_nt"] = eval_sample["gt_nt"]
        ref_result["pred_nt"] = ((pred_bbox["scores"] > threshold).sum() == 0).to("cpu")
        # No-target Samples
        if eval_sample["gt_nt"]:
            if ref_result["pred_nt"]:
                acc = 1
            else:
                acc = 0
            mean_acc.append(acc)
        # Targeted Samples
        else:
            I, U = computeIoU(pred_mask, gt_mask)
            if U == 0:
                this_iou = 0.0
            else:
                this_iou = I * 1.0 / U
            mean_IoU.append(this_iou)
            cum_I += I
            cum_U += U

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += this_iou >= eval_seg_iou

            seg_total += 1
    mIoU = np.mean(np.array(mean_IoU)) * 100.0
    macc = np.mean(np.array(mean_acc)) * 100.0
    oIoU = cum_I * 100.0 / cum_U

    for n_eval_iou in range(len(eval_seg_iou_list)):
        eval_seg_iou_list[n_eval_iou] = seg_correct[n_eval_iou] * 100.0 / seg_total

    return mIoU, oIoU, macc, eval_seg_iou_list
