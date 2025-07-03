import torch
from torch import nn
from torch.nn import functional as F
import torch
from deris.layers.box_ops import box_xyxy_to_cxcywh
from deris.models import HEADS
import pycocotools.mask as maskUtils
from deris.models.branchs.perception_branch.mask_config.config import get_mask_config
from deris.models.branchs.perception_branch.mask_criterion.Mask_Criterion import (
    HierVG_criterion,
    hungarian_matcher_HierVG,
)


@HEADS.register_module()
class SimpleHead_Independent_match(nn.Module):
    def __init__(
        self,
        loss_weight={"mask": 1.0, "bbox": 0.05, "existence": 1.0, "cls": 1.0, "aux": 0.1, "global_mask": 1.0},
        matching_cost_weight={"mask": 1.0, "bbox": 1.0, "cls": 1.0},
        mask_config=None,
        additional_detection_supervision={"enable": True, "loss_weight": 1.0, "box": 0.0, "mask": 1.0, "cls": 1.0},
        num_queries=20,
        stage_weights=[1.0, 1.0],
    ):
        super(SimpleHead_Independent_match, self).__init__()
        self.loss_weight_global = loss_weight
        self.matching_cost_weight = matching_cost_weight
        mask_decoder_cfg = get_mask_config(config=mask_config)
        self.mask_decoder_training_init(mask_decoder_cfg)
        self.additional_detection_supervision = additional_detection_supervision
        self.num_queries = num_queries
        self.stage_weights = stage_weights

    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        bbox_weight = 2.0
        boxgiou_weight = 2.0

        matcher = hungarian_matcher_HierVG(
            cost_class=class_weight * self.matching_cost_weight["cls"],
            cost_mask=mask_weight * self.matching_cost_weight["mask"],
            cost_dice=dice_weight * self.matching_cost_weight["mask"],
            cost_box=bbox_weight * self.matching_cost_weight["bbox"],
            cost_giou=boxgiou_weight * self.matching_cost_weight["bbox"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_class": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_bbox": bbox_weight,
            "loss_boxgiou": boxgiou_weight,
        }

        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v * self.loss_weight_global["aux"] for k, v in weight_dict.items()}
                )
            weight_dict.update(aux_weight_dict)

        losses = ["class", "masks", "boxes"]

        self.criterion = HierVG_criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        self.size_divisibility = 32

    def loss_add_weight(self, losses, device):
        loss_class = torch.tensor(0.0, device=device)
        loss_mask = torch.tensor(0.0, device=device)
        loss_det = torch.tensor(0.0, device=device)
        for k in list(losses.keys()):
            if k in self.weight_dict:
                if losses[k] is not None:
                    losses[k] *= self.weight_dict[k]
                if "class" in k and losses[k] is not None:
                    loss_class += losses[k]
                elif "_mask" in k or "_dice" in k:
                    loss_mask += losses[k]
                elif "_bbox" in k or "_giou" in k:
                    loss_det += losses[k]
            else:
                losses.pop(k)

        return loss_class, loss_mask, loss_det

    def prepare_refer_targets(self, targets, img_metas):
        new_targets = []
        is_empty = []
        refer_indices = [meta["refer_target_index"] for meta in img_metas]
        for ind, target_bbox, target_mask, img_meta in zip(
            refer_indices, targets["bbox"], targets["mask_parts"], img_metas
        ):
            if len(target_mask) != 0:
                gt_masks = torch.stack(
                    [torch.tensor(maskUtils.decode(m), device=target_bbox.device) for m in target_mask], dim=0
                )
            else:
                gt_masks = torch.zeros(
                    (0, img_meta["img_shape"][0], img_meta["img_shape"][1]), device=target_bbox.device
                )
            assert len(gt_masks) == len(target_bbox)
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox) == 0:
                target_bbox = torch.zeros((0, 4), device=target_bbox.device)
            else:
                target_bbox = target_bbox.reshape(-1, 4)
            if len(ind) == 0:
                is_empty.append(0)
            else:
                is_empty.append(1)
            gt_classes = torch.zeros(target_bbox.shape[0], device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes[ind], "boxes": gt_boxes[ind], "masks": gt_masks[ind]})
        is_empty = torch.tensor(is_empty, device=target_bbox.device).long()
        return new_targets, is_empty

    def forward_train(self, predictions, targets, perception_prediction=None):

        perp_loss = torch.tensor(0.0, device=predictions["pred_existence"].device)

        # prepare refer targets
        target_gt_refer, existence = self.prepare_refer_targets(targets, targets["img_metas"])
        refer_loss = self.criterion(predictions, target_gt_refer)
        loss_class, loss_mask, loss_det = self.loss_add_weight(refer_loss, device=predictions["pred_existence"].device)
        loss_class = self.loss_weight_global["cls"] * loss_class
        loss_mask = self.loss_weight_global["mask"] * loss_mask
        loss_det = self.loss_weight_global["bbox"] * loss_det

        # existence loss
        pred_existence = predictions["pred_existence"]
        existence_score = torch.sigmoid(pred_existence)
        loss_existence = F.binary_cross_entropy(
            existence_score.reshape(-1), existence.reshape(-1).float(), reduction="mean"
        )
        loss_existence = self.loss_weight_global["existence"] * loss_existence

        loss_dict = {
            "loss_class": loss_class,
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_label_nt": loss_existence,
            "loss_perp": perp_loss,
        }

        return loss_dict

    @torch.jit.unused
    def _set_aux_outputs(self, outputs_seg_masks, outputs_det_boxes, outputs_cls):
        return [
            {"pred_masks": c, "pred_boxes": b, "pred_logits": a}
            for c, b, a in zip(outputs_seg_masks[:-1], outputs_det_boxes[:-1], outputs_cls[:-1])
        ]
