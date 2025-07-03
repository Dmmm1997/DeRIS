from PIL import Image, ImageDraw, ImageFont
from deris.models.heads.uni_head_simple import get_maskouterbox
from deris.models.utils import xywh_to_x1y1x2y2
from deris.utils import is_main
import os

# from ..models.heads.uni_head_simple import get_maskouterbox
import cv2
import random
import pycocotools.mask as maskUtils
import numpy as np
import torch

font = ImageFont.load_default()


def box_seg_visualization(
    pred_box, pred_seg, pred_box_first, pred_seg_first, save_filename, img_metas, text, gt_mask, gt_box, threshold=0.5
):
    H, W = pred_seg.shape[-2:]

    gt_mask = maskUtils.decode(gt_mask)

    pred_box = (xywh_to_x1y1x2y2(pred_box).cpu().detach().numpy() * [W, H, W, H]).astype(np.int32)
    pred_seg = pred_seg.sigmoid().squeeze(0)
    pred_seg[pred_seg < threshold] = 0.0
    pred_seg[pred_seg >= threshold] = 1.0
    pred_segouterbox = get_maskouterbox(pred_seg.unsqueeze(0), threshold=threshold).squeeze(0)
    pred_seg = pred_seg.cpu().detach().numpy().astype(np.int32)

    pred_box_first = (xywh_to_x1y1x2y2(pred_box_first).cpu().detach().numpy() * [W, H, W, H]).astype(np.int32)
    pred_seg_first = pred_seg_first.sigmoid().squeeze(0)
    pred_seg_first[pred_seg_first < threshold] = 0.0
    pred_seg_first[pred_seg_first >= threshold] = 1.0
    pred_segouterbox_first = get_maskouterbox(pred_seg_first.unsqueeze(0), threshold=threshold).squeeze(0)
    pred_seg_first = pred_seg_first.cpu().detach().numpy().astype(np.int32)

    # draw pred
    mask_image = Image.fromarray(pred_seg * 255)
    image_pred = Image.new("RGB", (W, H))
    image_pred.paste(mask_image)
    draw_pred = ImageDraw.Draw(image_pred)
    draw_pred.rectangle(list(pred_box), outline="blue", width=2)
    # draw_pred.rectangle(list(pred_segouterbox), outline="blue", width=2)
    if isinstance(text, str):
        text_position = (10, 10)
        draw_pred.text(text_position, text, fill="red", font=font)

    # draw pred first
    mask_image_first = Image.fromarray(pred_seg_first * 255)
    image_pred_first = Image.new("RGB", (W, H))
    image_pred_first.paste(mask_image_first)
    draw_pred_first = ImageDraw.Draw(image_pred_first)
    draw_pred_first.rectangle(list(pred_box_first), outline="blue", width=2)
    # draw_pred_first.rectangle(list(pred_segouterbox_first), outline="blue", width=2)
    if isinstance(text, str):
        text_position = (10, 10)
        draw_pred_first.text(text_position, text, fill="red", font=font)

    # draw gt
    box_gt = (gt_box.cpu().detach().numpy()).astype(np.int32)
    mask_gt = gt_mask.astype(np.int32)
    mask_gt = Image.fromarray(mask_gt * 255)
    image_gt = Image.new("RGB", (W, H))
    image_gt.paste(mask_gt)
    draw_gt = ImageDraw.Draw(image_gt)
    draw_gt.rectangle(list(box_gt), outline="red", width=2)

    # draw source image
    img_metas["filename"]
    file_name = img_metas["filename"]
    expression = img_metas["expression"]
    img_source = Image.open(file_name)
    img_source = img_source.resize((W, H))

    concat_image = Image.new("RGB", (W * 4 + 30, H), "white")
    concat_image.paste(img_source, (0, 0))
    concat_image.paste(image_gt, (W + 10, 0))
    concat_image.paste(image_pred_first, (W * 2 + 20, 0))
    concat_image.paste(image_pred, (W * 3 + 30, 0))

    save_filename = save_filename + "-{}-{}".format(expression, file_name.split("/")[-1])

    concat_image.save(save_filename)


def heatmap_visulization(featmap, saved_path):
    heatmap = featmap.cpu().detach().numpy()[0]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(saved_path, colored_heatmap)


def visualize_attention_with_image(image, attention, save_path=None, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    使用 OpenCV 可视化注意力图，并叠加到原始图像上。

    参数:
    - image: (np.ndarray) 原始图像，形状为 (H, W, 3)。
    - attention: (torch.Tensor or np.ndarray) 注意力权重矩阵，形状为 (H, W)。
    - save_path: (str) 如果提供路径，则保存可视化图片。
    - alpha: (float) 热力图与原始图像的融合比例，默认 0.6。
    - colormap: (int) 使用的 OpenCV 颜色映射，默认使用 COLORMAP_JET。

    输出:
    - 显示叠加了注意力图的图像。
    """

    # 如果 attention 是 torch.Tensor，转换为 numpy 数组
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    # 检查注意力图的大小是否与图像相匹配
    H, W = image.shape[:2]
    attention_resized = cv2.resize(attention, (W, H))

    # 归一化注意力图到 [0, 255]
    attention_normalized = cv2.normalize(attention_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    attention_normalized = attention_normalized.astype(np.uint8)

    # 将注意力图转换为彩色热力图
    heatmap = cv2.applyColorMap(attention_normalized, colormap)

    # 将热力图叠加到原始图像上
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    # 显示叠加结果
    cv2.imshow("Attention Map Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 如果提供了保存路径，保存图像
    if save_path:
        cv2.imwrite(save_path, overlay)
