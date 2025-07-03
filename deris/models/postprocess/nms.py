import torch

def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes using PyTorch.

    Parameters:
    - boxes: A torch tensor of shape (N, 4) where N is the number of boxes and each box is represented by [x1, y1, x2, y2].
    - scores: A torch tensor of shape (N,) containing the scores for each box.
    - iou_threshold: A float representing the IoU threshold for suppressing boxes.

    Returns:
    - A list of indices of the boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    # Compute the area of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort the boxes by their scores in descending order
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute the intersection areas
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h

        # Compute the IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU less than the threshold
        order = order[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.int64)
