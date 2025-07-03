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


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def gres_evaluate(pred_bboxes, pred_masks, gt_masks, threshold=0.7):
    """
    prediction should have the result of pred_nt, gt_nt
    """

    def computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U

    pr_thres = [0.7, 0.8, 0.9]

    accum_I, accum_U, accum_IoU, total_count, not_empty_count, empty_count = 0, 0, 0, 0, 0, 0
    pr_count = defaultdict(int)
    nt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    results_dict = []

    for ind, (pred_bbox, eval_sample) in enumerate(zip(pred_bboxes, pred_masks)):
        pred_mask = maskUtils.decode(eval_sample["pred_masks"])[None]
        gt_mask = maskUtils.decode(gt_masks[ind])[None]

        # eval_sample should have the key of ["gt_nt", "pred_nt", "gt_mask", "pred_mask"]
        ref_result = {}
        ref_result["gt_nt"] = eval_sample["gt_nt"]
        ref_result["pred_nt"] = ((pred_bbox["scores"] > threshold).sum() == 0).to("cpu")
        # ref_result["pred_nt"] = (eval_sample["pred_nt"] > 0.5).bool().to("cpu")

        # if (eval_sample["pred_nt"]>0.5).bool().to("cpu") != ((pred_bbox["scores"] > threshold).sum() == 0).to("cpu"):
        #     print(1)

        I, U = computeIoU(pred_mask, gt_mask)

        # No-target Samples
        if eval_sample["gt_nt"]:
            empty_count += 1
            ref_result["I"] = int(0)

            # True Positive
            if ref_result["pred_nt"]:
                nt["TP"] += 1
                accum_IoU += 1
                accum_I += 0
                accum_U += 0

                ref_result["U"] = int(0)
                ref_result["cIoU"] = float(1)

            # False Negative
            else:
                nt["FN"] += 1
                accum_IoU += 0
                accum_I += 0
                accum_U += int(U)

                ref_result["U"] = int(U)
                ref_result["cIoU"] = float(0)

        # Targeted Samples
        else:
            # False Positive
            if ref_result["pred_nt"]:
                nt["FP"] += 1
                I = 0

            # True Negative
            else:
                nt["TN"] += 1

            this_iou = float(0) if U == 0 else float(I) / float(U)

            accum_IoU += this_iou
            accum_I += I
            accum_U += U

            not_empty_count += 1

            for thres in pr_thres:
                if this_iou >= thres:
                    pr_count[thres] += 1

            ref_result["I"] = int(I)
            ref_result["U"] = int(U)
            ref_result["cIoU"] = float(this_iou)

        total_count += 1
        results_dict.append(ref_result)

    res = {}
    res["gIoU"] = 100.0 * (accum_IoU / total_count)
    res["cIoU"] = accum_I * 100.0 / accum_U

    if empty_count > 0:
        res["T_acc"] = (nt["TN"] / (nt["TN"] + nt["FP"])) * 100
        res["N_acc"] = (nt["TP"] / (nt["TP"] + nt["FN"])) * 100
    else:
        res["T_acc"] = res["N_acc"] = 0

    for thres in pr_thres:
        pr_name = "Pr@{0:1.1f}".format(thres)
        res[pr_name] = pr_count[thres] * 100.0 / not_empty_count

    return res["gIoU"], res["cIoU"], res["T_acc"], res["N_acc"], [res["Pr@0.7"], res["Pr@0.8"], res["Pr@0.9"]]


def grec_evaluate_f1_nacc(predictions, gt_bboxes, img_metas, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0):
    count = 0
    correct_image = 0
    num_image = 0
    nt = {
        "TP": 0.0,
        "TN": 0.0,
        "FP": 0.0,
        "FN": 0.0,
    }
    if predictions is None:
        return 0.0, 0.0
    for prediction, gt_bbox, target in zip(predictions, gt_bboxes, img_metas):
        TP = 0
        assert prediction is not None
        if "filtered_scores" in prediction:
            sorted_scores_boxes = sorted(
                zip(prediction["filtered_scores"].tolist(), prediction["filtered_boxes"].tolist()), reverse=True
            )
        else:
            sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
        sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
        no_target_flag = False
        if target["empty"]:
            no_target_flag = True
        gt_bbox_all = gt_bbox
        sorted_scores_array = numpy.array(sorted_scores)
        idx = sorted_scores_array > thresh_score
        filtered_boxes = sorted_boxes[idx]

        # if prediction["pred_nt"]>0.5:
        #     filtered_boxes = torch.zeros([0,4])
        giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4))
        num_prediction = filtered_boxes.shape[0]
        num_gt = gt_bbox_all.shape[0]

        if no_target_flag:
            if num_prediction >= 1:
                nt["FN"] += 1
                F_1 = 0.0
            else:
                nt["TP"] += 1
                F_1 = 1.0
        else:
            if num_prediction >= 1:
                nt["TN"] += 1
            else:
                nt["FP"] += 1
            for i in range(min(num_prediction, num_gt)):
                top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                if top_value < thresh_iou:
                    break
                else:
                    top_index_x = top_index // num_gt
                    top_index_y = top_index % num_gt
                    TP += 1
                    giou[top_index_x[0], :] = 0.0
                    giou[:, top_index_y[0]] = 0.0
            FP = num_prediction - TP
            FN = num_gt - TP
            F_1 = 2 * TP / (2 * TP + FP + FN)
            if F_1 == 1:
                count = count + 1

        if F_1 >= thresh_F1:
            correct_image += 1
        num_image += 1
    # return {"correct_image": correct_image, "num_image": num_image}, nt
    # print('nt["FN"] ',nt["FN"] )
    # print('nt["TP"] ',nt["TP"] )
    # print('nt["TN"] ',nt["TN"] )
    # print('nt["FP"] ',nt["FP"] )
    # print('count',count)
    # print('correct_image',correct_image)
    # print('num_image',num_image)

    F1_score = correct_image / num_image
    T_acc = nt["TN"] / (nt["TN"] + nt["FP"]) if nt["TN"] != 0 else 0.0
    N_acc = nt["TP"] / (nt["TP"] + nt["FN"]) if nt["TP"] != 0 else 0.0
    return F1_score * 100, N_acc * 100, T_acc * 100
