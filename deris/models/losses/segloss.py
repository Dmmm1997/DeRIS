import torch
from torch.nn import functional as F
import torch


def refer_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor):
    loss = F.cross_entropy(inputs, targets, weight=weight)
    return loss


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def sigmoid_ce_loss(inputs, targets):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    return ce_loss


def seg_loss(inputs, target, loss_info):
    # weight = torch.FloatTensor([0.9, 1.1]).cuda()
    # return loss_func(inputs, target.long(), weight=weight)
    loss_seg = torch.tensor([0.0], device=inputs.device)
    target = target.float().unsqueeze(1)
    assert target.shape == inputs.shape
    if "dice" in loss_info:
        loss_seg += dice_loss(inputs, target) * loss_info["dice"]
    if "bce" in loss_info:
        loss_seg += sigmoid_ce_loss(inputs, target) * loss_info["bce"]

    return loss_seg


def part_seg_loss(inputs, targets, indices, loss_info):
    loss_seg = 0.0
    total_pred_masks_pos = []
    total_gt_masks_pos = []
    total_pred_masks_neg = []
    total_gt_mask_neg = []
    for pred_mask, gt_part_mask, indice in zip(inputs, targets, indices[-1]):
        neg_mask = torch.ones(pred_mask.size(0), dtype=torch.bool)
        neg_mask[indice[0]] = False
        pred_mask_neg = pred_mask[neg_mask]
        gt_masks_neg = torch.zeros(
            (pred_mask.shape[0] - len(indice[0]), pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float
        ).to(pred_mask.device)
        total_pred_masks_neg.append(pred_mask_neg)
        total_gt_mask_neg.append(gt_masks_neg)
        if len(indice) == 0:
            total_pred_masks_pos.append(
                torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device)
            )
            total_gt_masks_pos.append(
                torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device)
            )
            continue
        pred_mask_pos = pred_mask[indice[0]]
        gt_mask_pos = torch.tensor(gt_part_mask[indice[1].cpu().detach().numpy().tolist()]).to(pred_mask.device)
        total_pred_masks_pos.append(pred_mask_pos)
        total_gt_masks_pos.append(gt_mask_pos)
    pred_masks_pos = torch.concat(total_pred_masks_pos, dim=0)
    pred_mask_neg = torch.concat(total_pred_masks_neg, dim=0)
    gt_masks_pos = torch.concat(total_gt_masks_pos, dim=0)
    gt_masks_neg = torch.concat(total_gt_mask_neg, dim=0)
    assert pred_masks_pos.shape == gt_masks_pos.shape
    assert pred_mask_neg.shape == gt_masks_neg.shape
    loss_seg_pos = seg_loss(pred_masks_pos.unsqueeze(1), gt_masks_pos, loss_info)
    loss_seg_neg = seg_loss(pred_mask_neg.unsqueeze(1), gt_masks_neg, loss_info) * loss_info["neg"]
    loss_seg = loss_seg_pos + loss_seg_neg
    return loss_seg


def refer_instance_seg_loss(inputs, targets, indices, refer_indices, loss_info):
    loss_seg = 0.0
    total_pred_masks_pos = []
    total_gt_masks_pos = []
    total_pred_masks_neg = []
    total_gt_mask_neg = []
    for pred_mask, gt_part_mask, indice, refer_indice in zip(inputs, targets, indices[-1], refer_indices):
        new_index = [i for i, index in enumerate(indice[0]) if index in refer_indice]
        new_indice = [indice[0][new_index], indice[1][new_index]]

        neg_mask = torch.ones(pred_mask.size(0), dtype=torch.bool)
        neg_mask[new_indice[0]] = False
        pred_mask_neg = pred_mask[neg_mask]
        gt_masks_neg = torch.zeros(
            (pred_mask.shape[0] - len(new_indice[0]), pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float
        ).to(pred_mask.device)
        total_pred_masks_neg.append(pred_mask_neg)
        total_gt_mask_neg.append(gt_masks_neg)
        if len(new_indice) == 0:
            total_pred_masks_pos.append(
                torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device)
            )
            total_gt_masks_pos.append(
                torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device)
            )
            continue
        pred_mask_pos = pred_mask[new_indice[0]]
        gt_mask_pos = torch.tensor(gt_part_mask[new_indice[1].cpu().detach().numpy().tolist()]).to(pred_mask.device)
        total_pred_masks_pos.append(pred_mask_pos)
        total_gt_masks_pos.append(gt_mask_pos)
    pred_masks_pos = torch.concat(total_pred_masks_pos, dim=0)
    pred_mask_neg = torch.concat(total_pred_masks_neg, dim=0)
    gt_masks_pos = torch.concat(total_gt_masks_pos, dim=0)
    gt_masks_neg = torch.concat(total_gt_mask_neg, dim=0)
    assert pred_masks_pos.shape == gt_masks_pos.shape
    assert pred_mask_neg.shape == gt_masks_neg.shape
    loss_seg_pos = seg_loss(pred_masks_pos.unsqueeze(1), gt_masks_pos, loss_info)
    loss_seg_neg = seg_loss(pred_mask_neg.unsqueeze(1), gt_masks_neg, loss_info) * loss_info["neg"]
    loss_seg = loss_seg_pos + loss_seg_neg
    return loss_seg


def FocalLoss(inputs, targets, alpha=0.25, gamma=2, reduction="mean"):

    # 将 logits 转换为概率
    probs = F.softmax(inputs, dim=-1)  # 形状: [B, N, 2]

    # # 获取目标类别的概率
    # class_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # 形状: [B, N]
    # # 计算 Focal Loss
    # focal_loss = -alpha * (1 - class_probs).pow(gamma) * torch.log(class_probs + 1e-6)

    # 获取目标类别的概率
    # 注意：targets 为 0 表示正类，1 表示负类
    class_probs = probs.gather(-1, (1 - targets).unsqueeze(-1)).squeeze(-1)  # 形状: [B, N]
    # 计算 Focal Loss
    focal_loss = -alpha * (1 - class_probs).pow(gamma) * torch.log(class_probs + 1e-6)

    # 根据 reduction 参数返回损失
    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss
