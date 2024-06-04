# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import cv2
import numpy as np
import random
import torch, os
from torchvision.ops.boxes import box_area


def bbox_xyxy_to_cxcyah(bboxes):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, ratio, h).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    return xyah


def bbox_cxcyah_to_xyxy(bboxes):
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, x2, y2).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    return torch.cat(x1y1x2y2, dim=-1)


def random_color(ins_id):
    random.seed(ins_id)
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def visualize_boxes(
    img,
    boxes,
    save_path,
    height=None,
    width=None,
    score=None,
    boxes2=None,
    score2=None,
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if isinstance(img, torch.Tensor):
        # the expected shape is (3, H, W)
        img = img.cpu().numpy()

        for i, (m, s) in enumerate(zip(mean, std)):
            img[i] = img[i] * s + m

        img = np.transpose(img, (1, 2, 0))
        img = img * 255
        img = img.astype(np.uint8)

        # convert to BGR
        img = img[:, :, ::-1].copy()

    elif isinstance(img, str):
        img = cv2.imread(img)
        # img = img[:, :, ::-1].copy()

    if boxes is None:
        cv2.imwrite(save_path, img)
        return

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.FloatTensor(boxes)

    if height is not None:
        try:
            boxes[:, 1::2] *= height
            boxes[:, 0::2] *= width

            boxes = boxes.cpu().numpy()
        except:
            print("Error in visualizing boxes")
            print(boxes.shape)
            return

    src = img.copy()
    for ins_id, insb in enumerate(boxes):
        # print(insb)
        vis_color = np.array(random_color(ins_id)).astype(np.int32)
        color_rgb = vis_color.tolist()
        color_rgb = [int(c) for c in color_rgb]

        cv2.rectangle(
            src,
            (
                int(insb[0]),
                int(insb[1]),
            ),
            (
                int(insb[2]),
                int(insb[3]),
            ),
            tuple(color_rgb),
            6,
        )

        if boxes2 is not None:
            insb2 = boxes2[ins_id]
            cv2.rectangle(
                src,
                (
                    int(insb2[0]),
                    int(insb2[1]),
                ),
                (
                    int(insb2[2]),
                    int(insb2[3]),
                ),
                (0, 255, 255),
                6,
            )

    if score is not None:
        # put text to the image
        cv2.putText(
            src,
            f"{score:.4f}",
            (int(boxes[0][0]), int(boxes[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if score2 is not None:
        # put text to the image
        cv2.putText(
            src,
            f"{score2:.4f}",
            (int(boxes2[0][0]), int(boxes2[0][1] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        save_path,
        src,
    )
    # print(f"Saved to {save_path}")
    return src


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
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

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)  # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def scale_boxes(boxes, scale):
    """
    Args:
        boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
            coords representing the corners x0, y0, x1, y1,
        scale (float): The box scaling factor.

    Returns:
        Scaled boxes.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_boxes = torch.zeros_like(boxes)
    scaled_boxes[:, 0] = x_c - w_half
    scaled_boxes[:, 2] = x_c + w_half
    scaled_boxes[:, 1] = y_c - h_half
    scaled_boxes[:, 3] = y_c + h_half
    return scaled_boxes


if __name__ == "__main__":
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb

    ipdb.set_trace()
