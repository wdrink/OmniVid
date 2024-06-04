import os
import glob
import pickle
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
import torch.nn as nn
from torchvision.ops.boxes import nms
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url
from od_util import box_ops

from lavis.models.actionformer_models.utils import batched_nms

"""
Copied from MMAction2
https://github.com/open-mmlab/mmaction2/blob/master/mmaction/core/evaluation/eval_detection.py
"""
import json
from sklearn.metrics import precision_recall_curve
from collections import OrderedDict, defaultdict
import time
import copy
import multiprocessing as mp


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]

        return results


def iou(pred, gt):  # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(
        gt, list
    ), f"pred is {pred}, and gt is {gt}."
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def display_results(eval_result, miou, tious, recalls, title=None):
    # tious = [0.3, 0.5, 0.7]
    # recalls = [1, 5]

    display_data = [
        ["Rank@{},mIoU@{}".format(i, j) for i in recalls for j in tious]
        + ["mIoU"]
    ]
    eval_result = eval_result * 100
    miou = miou * 100
    display_data.append(
        [
            "{:.02f}".format(eval_result[j][i])
            for i in range(len(recalls))
            for j in range(len(tious))
        ]
        + ["{:.02f}".format(miou)]
    )
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = "center"
    return table.table


def eval(segments, gt_timestamps, scores=None):
    """Evaluate the predicted segments with ground truth timestamps.
    Args:
        segments (list[list[float]]): Predicted segments.
        gt_timestamps (list[list[float]]): Ground truth timestamps.
        scores (list[float], optional): The confidence scores of predicted
            segments. Defaults to None.
    Returns:
        tuple[list[float]]: The evaluation results.
    """

    if scores is None:
        # only 1 predicted segment for each video
        tious = [0.3, 0.5, 0.7]
        recalls = [1]

        eval_result = [[[] for _ in recalls] for _ in tious]
        max_recall = max(recalls)
        average_iou = []

        for seg, dat in zip(segments, gt_timestamps):
            overlap = iou(seg, dat)
            average_iou.append(np.mean(np.sort(overlap)[-3:]))

            for i, t in enumerate(tious):
                for j, r in enumerate(recalls):
                    eval_result[i][j].append((overlap > t)[:r].any())

        eval_result = np.array(eval_result).mean(axis=-1)
        miou = np.mean(average_iou)

    else:
        # multiple predicted segments for each video
        tious = [0.3, 0.5, 0.7]
        recalls = [1, 5]

        eval_result = [[[] for _ in recalls] for _ in tious]
        max_recall = max(recalls)
        average_iou = []

        for seg, dat, score in zip(segments, gt_timestamps, scores):
            # we first sort the segments by scores
            seg = np.array(seg)
            score = np.array(score)
            sorted_idx = np.argsort(score)[::-1]
            seg = seg[sorted_idx].tolist()
            score = score[sorted_idx].tolist()

            if len(seg) == 0:
                seg = [[0, 0]]
                score = [0]

            if len(seg) < max_recall:
                seg = seg + [seg[-1]] * (max_recall - len(seg))
                score = score + [score[-1]] * (max_recall - len(score))

            overlap = iou(seg, dat)
            average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

            for i, t in enumerate(tious):
                for j, r in enumerate(recalls):
                    eval_result[i][j].append((overlap > t)[:r].any())

        eval_result = np.array(eval_result).mean(axis=-1)
        miou = np.mean(average_iou)

    print(display_results(eval_result, miou, tious, recalls, ""))

    return eval_result


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """compute intersection-over-union along temporal axis for each pair of windows in pred_windows and gt_windows.
    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(
        0,
        np.minimum(pred_windows[:, 1], gt_windows[:, 1])
        - np.maximum(pred_windows[:, 0], gt_windows[:, 0]),
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) - np.minimum(
        pred_windows[:, 0], gt_windows[:, 0]
    )  # not the correct union though
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union != 0
    )


def compute_temporal_iou_batch_cross(spans1, spans2):
    """
    Args:
        spans1: (N, 2) np.ndarray, each row defines a span [st, ed]
        spans2: (M, 2) np.ndarray, ...

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    >>> spans1 = np.array([[0, 0.2, 0.9], [0.5, 1.0, 0.2]])
    >>> spans2 = np.array([[0, 0.3], [0., 1.0]])
    >>> compute_temporal_iou_batch_cross(spans1, spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x["score"])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_videoid.setdefault(item["video-id"], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred["video-id"] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred["video-id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array(
            [
                [pred["t-start"], pred["t-end"]],
            ]
        )
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(
            precision_cumsum[t_idx, :], recall_cumsum[t_idx, :]
        )
    return ap


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision

    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    """
    # Check inputs
    assert len(y_true) == len(
        y_predict
    ), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # True labels are all zeros
            # raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [
            0,
            1,
        ], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [
            precision[np.where(recall >= t)[0][-1]]
            for t in np.arange(0, 1.01, 0.1)
        ]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])


def compute_average_precision_detection_wrapper(
    input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_mr_ap(
    submission,
    ground_truth,
    iou_thds=np.linspace(0.5, 0.95, 10),
    max_gt_windows=None,
    max_pred_windows=10,
    num_workers=8,
    chunksize=50,
):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = (
            d["pred_relevant_windows"][:max_pred_windows]
            if max_pred_windows is not None
            else d["pred_relevant_windows"]
        )
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append(
                {
                    "video-id": d["qid"],  # in order to use the API
                    "t-start": w[0],
                    "t-end": w[1],
                    "score": w[2],
                }
            )

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = (
            d["relevant_windows"][:max_gt_windows]
            if max_gt_windows is not None
            else d["relevant_windows"]
        )
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append(
                {"video-id": d["qid"], "t-start": w[0], "t-end": w[1]}
            )
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [
        [qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data
    ]
    from functools import partial

    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds
    )

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(
                compute_ap_from_triple, data_triples, chunksize=chunksize
            ):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(
    submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)
):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {
        d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission
    }  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if (
            len(cur_gt_windows) > 0
        ):  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]),
                np.array(d["relevant_windows"]),
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(
            f"{np.mean(pred_gt_iou >= thd) * 100:.2f}"
        )
    return iou_thd2recall_at_one


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    """keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w
            for w in d["relevant_windows"]
            if min_l < get_window_len(w) <= max_l
        ]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    length_ranges = [
        [0, 10],
        [10, 30],
        [30, 150],
        [0, 150],
    ]  #
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(
            submission, ground_truth, l_range
        )
        print(
            f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
            f"{100*len(_ground_truth)/len(ground_truth):.2f} examples."
        )
        iou_thd2average_precision = compute_mr_ap(
            _submission, _ground_truth, num_workers=8, chunksize=50
        )
        iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
        ret_metrics[name] = {
            "MR-mAP": iou_thd2average_precision,
            "MR-R1": iou_thd2recall_at_one,
        }
        if verbose:
            print(
                f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds"
            )
    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {
        k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()
    }
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]  # (#clips, 3)
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    # aggregate scores from 3 separate annotations (3 workers) by taking the max.
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {
        k: v["pred_saliency_scores"] for k, v in qid2preds.items()
    }
    ap_scores = np.zeros((len(qid2preds), 3))  # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                compute_ap_from_tuple, input_tuples, chunksize=chunksize
            ):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        # print(f"len(y_true) < len(y_predict) {len(y_true), len(y_predict)}")
        y_predict = y_predict[: len(y_true)]
    elif len(y_true) > len(y_predict):
        # print(f"len(y_true) > len(y_predict) {len(y_true), len(y_predict)}")
        _y_predict = np.zeros(len(y_true))
        _y_predict[: len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    """gt_data, dict,"""
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    relevant_clip_ids = np.array(
        gt_data["relevant_clip_ids"]
    )  # (#relevant_clip_ids, )
    saliency_scores_relevant_clips = np.array(
        gt_data["saliency_scores"]
    )  # (#relevant_clip_ids, 3)
    saliency_scores_full_video[
        relevant_clip_ids
    ] = saliency_scores_relevant_clips
    return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4]


def eval_highlight(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    qid2preds = {d["qid"]: d for d in submission}
    qid2gt_scores_full_range = {
        d["qid"]: mk_gt_scores(d) for d in ground_truth
    }  # scores in range [0, 4]
    # gt_saliency_score_min: int, in [0, 1, 2, 3, 4]. The minimum score for a positive clip.
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(
        gt_saliency_score_min_list, saliency_score_names
    ):
        start_time = time.time()
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()
        }  # scores in [0, 1]
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {
            "HL-mAP": mean_ap,
            "HL-Hit1": hit_at_one,
        }
        if verbose:
            print(
                f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})"
            )
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics


def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    if isinstance(ground_truth, str):
        ground_truth = load_jsonl(ground_truth)

    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, (
            f"qids in ground_truth and submission must match. "
            f"use `match_number=False` if you wish to disable this check"
        )
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose
        )
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
        }
        eval_metrics_brief.update(
            sorted(
                [(k, v) for k, v in moment_ret_scores_brief.items()],
                key=lambda x: x[0],
            )
        )

    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(
            submission, ground_truth, verbose=verbose
        )
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict(
            [
                (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
                for k, v in highlight_det_scores.items()
                for sub_k in v
            ]
        )
        eval_metrics_brief.update(highlight_det_scores_brief)

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(
        sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0])
    )
    return final_eval_metrics


def evaluate(anno_root, pred_root):
    pred_files = os.listdir(pred_root)
    otb_files_subm = []
    otb_files_anno = []
    for pred_txt in pred_files:
        if not ".txt" in pred_txt:
            continue
        otb_files_subm.append(os.path.join(pred_root, pred_txt))
        otb_files_anno.append(os.path.join(anno_root, "anno", pred_txt))

    # pandaframe with all the OPE results
    df_all = pd.DataFrame(
        columns=["sanity_check", "IoU", "distance", "ndistance"]
    )
    n = 21
    Success_test_avg = np.zeros(n)
    Precision_test_avg = np.zeros(n)
    NPrecision_test_avg = np.zeros(n)
    Success_X = np.linspace(0, 1, n)
    Precision_X = np.linspace(0, 50, n)
    NPrecision_X = np.linspace(0, 0.5, n)

    """onebyone_success = 0.
    onebyone_precision = 0.
    onebyone_nprecision = 0."""

    per_vid_re = dict()

    # loop over the annotation files
    for i, (name_file_anno, name_file_subm) in tqdm(
        enumerate(zip(otb_files_anno, otb_files_subm))
    ):
        file = open(name_file_anno)
        df = pd.read_csv(
            file, sep=",", names=["anno_x1", "anno_y1", "anno_w", "anno_h"]
        )

        file_subm = open(name_file_subm)
        df_subm = pd.read_csv(
            file_subm, sep=",", names=["subm_x1", "subm_y1", "subm_w", "subm_h"]
        )
        df["sanity_check"] = 1.0

        # copy submission to dataframe
        df["subm_x1"] = df_subm["subm_x1"]
        df["subm_y1"] = df_subm["subm_y1"]
        df["subm_w"] = df_subm["subm_w"]
        df["subm_h"] = df_subm["subm_h"]
        df["subm_w"] = np.round(df_subm["subm_w"].astype(float))
        df["subm_h"] = np.round(df_subm["subm_h"].astype(float))
        df["anno_w"] = np.round(df["anno_w"].astype(float))
        df["anno_h"] = np.round(df["anno_h"].astype(float))
        df["subm_x1"] = np.round(df_subm["subm_x1"].astype(float))
        df["subm_y1"] = np.round(df_subm["subm_y1"].astype(float))
        df["anno_x1"] = np.round(df["anno_x1"].astype(float))
        df["anno_y1"] = np.round(df["anno_y1"].astype(float))

        df.loc[0, "subm_x1"] = df.loc[0, "anno_x1"]
        df.loc[0, "subm_y1"] = df.loc[0, "anno_y1"]
        df.loc[0, "subm_w"] = df.loc[0, "anno_w"]
        df.loc[0, "subm_h"] = df.loc[0, "anno_h"]

        # from (x,y,w,h) to (x1,y1,x2,y2)
        df["anno_x2"] = df["anno_x1"] + df["anno_w"] - 1.0
        df["anno_y2"] = df["anno_y1"] + df["anno_h"] - 1.0
        df["subm_x2"] = df["subm_x1"] + df["subm_w"] - 1.0
        df["subm_y2"] = df["subm_y1"] + df["subm_h"] - 1.0

        # compute centers for BB
        df["anno_center_x"] = df["anno_x1"] + (df["anno_w"] + 1.0) / 2.0
        df["anno_center_y"] = df["anno_y1"] + (df["anno_h"] + 1.0) / 2.0
        df["subm_center_x"] = df["subm_x1"] + (df["subm_w"] + 1.0) / 2.0
        df["subm_center_y"] = df["subm_y1"] + (df["subm_h"] + 1.0) / 2.0

        # compute (x1,y1,x2,y2) of interection
        df["inter_x1"] = df[["anno_x1", "subm_x1"]].max(axis=1)
        df["inter_y1"] = df[["anno_y1", "subm_y1"]].max(axis=1)
        df["inter_x2"] = df[["anno_x2", "subm_x2"]].min(axis=1)
        df["inter_y2"] = df[["anno_y2", "subm_y2"]].min(axis=1)

        # compute the area of intersection rectangle
        df["inter_w"] = np.round(df["inter_x2"] - df["inter_x1"] + 1).clip(
            lower=0
        )
        df["inter_h"] = np.round(df["inter_y2"] - df["inter_y1"] + 1).clip(
            lower=0
        )
        df["inter_area"] = df["inter_w"] * df["inter_h"]

        df["subm_area"] = (df["subm_h"]) * (df["subm_w"])
        df["anno_area"] = (df["anno_h"]) * (df["anno_w"])

        df["sanity_check"] = (
            df["subm_h"] + df["subm_w"] + df["subm_x1"] + df["subm_y1"]
        ) > 0

        df_all = df_all._append(df[["sanity_check"]])

        df.loc[df["sanity_check"] > 0, "IoU"] = -1.0

        # compute IoU
        df["IoU"] = df["inter_area"] / (
            df["anno_area"] + df["subm_area"] - df["inter_area"]
        )

        # compute center distance
        df["distance"] = (
            (df["anno_center_x"] - df["subm_center_x"])
            * (df["anno_center_x"] - df["subm_center_x"])
            + (df["anno_center_y"] - df["subm_center_y"])
            * (df["anno_center_y"] - df["subm_center_y"])
        ).apply(np.sqrt)

        # compute center distance normalized over the GT BB dimension
        df["ndistance"] = (
            (df["anno_center_x"] - df["subm_center_x"])
            / df["anno_w"]
            * (df["anno_center_x"] - df["subm_center_x"])
            / df["anno_w"]
            + (df["anno_center_y"] - df["subm_center_y"])
            / df["anno_h"]
            * (df["anno_center_y"] - df["subm_center_y"])
            / df["anno_h"]
        ).apply(np.sqrt)

        this_success = np.array(
            [
                np.sum(i >= thres for i in df["IoU"]).astype(float)
                / (len(df["IoU"]))
                for thres in Success_X
            ]
        )
        tmp_success = np.trapz(this_success, x=Success_X) * 100
        Success_test_avg += this_success

        this_precision = np.array(
            [
                np.sum(i <= thres for i in df["distance"]).astype(float)
                / (len(df["distance"]))
                for thres in Precision_X
            ]
        )
        tmp_precision = np.trapz(this_precision, x=Success_X) * 100
        Precision_test_avg += this_precision

        this_nprecision = np.array(
            [
                np.sum(i <= thres for i in df["ndistance"]).astype(float)
                / (len(df["ndistance"]))
                for thres in NPrecision_X
            ]
        )
        tmp_nprecision = np.trapz(this_nprecision, x=Success_X) * 100
        NPrecision_test_avg += this_nprecision

        # print(name_file_anno, tmp_success, tmp_precision, tmp_nprecision)
        """onebyone_success += tmp_success
        onebyone_precision += tmp_precision
        onebyone_nprecision += tmp_nprecision"""
        vid_name = os.path.basename(name_file_anno)
        per_vid_re[vid_name] = [tmp_success, tmp_precision, tmp_nprecision]

        # print(df.loc[df["IoU"] < 1.0])

    df_all.reset_index(drop=True, inplace=True)

    sanity_check_list = list(df_all["sanity_check"])

    # print(len(otb_files_anno))
    Precision = np.array(Precision_test_avg) / len(otb_files_anno)
    NPrecision = np.array(NPrecision_test_avg) / len(otb_files_anno)
    Success = np.array(Success_test_avg) / len(otb_files_anno)

    Sanity_check_Average = np.mean(sanity_check_list) * 100
    Success_Average = np.trapz(Success, x=Success_X) * 100
    Precision_Average = np.trapz(Precision, x=Success_X) * 100
    NPrecision_Average = np.trapz(NPrecision, x=Success_X) * 100

    return (
        Sanity_check_Average,
        Success_Average,
        Precision_Average,
        NPrecision_Average,
        per_vid_re,
    )


def visualize_video(frames, save_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # video inputs: T, 3, H, W
    imgs_vis = []
    for img in frames:
        # the expected shape is (3, H, W)
        img = img.cpu().numpy()

        for i, (m, s) in enumerate(zip(mean, std)):
            img[i] = img[i] * s + m

        img = np.transpose(img, (1, 2, 0))
        img = img * 255
        img = img.astype(np.uint8)

        # convert to BGR
        img = img[:, :, ::-1].copy()

        imgs_vis.append(img)

    # concat frames for visualization
    imgs_vis = np.concatenate(imgs_vis, axis=1)
    # print(save_path, imgs_vis.shape)
    # save the visualization
    cv2.imwrite(save_path, imgs_vis)
    print(f"save to {save_path}")
    return


def eval_at_thres(gt_dict, pred_dict, threshold):
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for vid_id in list(gt_dict.keys()):
        # filter by avg_f1 score
        if gt_dict[vid_id]["f1_consis_avg"] < 0.3:
            continue

        if vid_id not in pred_dict.keys():
            num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
            continue

        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]["fps"]
        my_dur = gt_dict[vid_id]["video_duration"]
        ins_start = 0
        ins_end = my_dur

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id][
            "substages_timestamps"
        ]
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros(
                (len(bdy_timestamps_list_gt), len(bdy_timestamps_det))
            )

            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(
                        bdy_timestamps_list_gt[ann1_idx]
                        - bdy_timestamps_det[ann2_idx]
                    )
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * my_dur:
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)

    return f1, rec, prec


def eval_gebd(pred_file, gt_file):
    # load GT files
    with open(gt_file, "rb") as f:
        gt_dict = pickle.load(f, encoding="lartin1")

    # load output files
    with open(pred_file, "rb") as f:
        pred_dict = pickle.load(f, encoding="lartin1")

    # recall precision f1 for threshold 0.05(5%)
    threshold = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    re_dict = {"threshold": threshold, "f1": [], "recall": [], "precision": []}

    for thres in threshold:
        f1, rec, prec = eval_at_thres(gt_dict, pred_dict, thres)
        re_dict["f1"].append(f1)
        re_dict["recall"].append(rec)
        re_dict["precision"].append(prec)

    return re_dict
