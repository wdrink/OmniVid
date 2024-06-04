"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import warnings

warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import math
import json
import librosa

import torch
import torch.nn.functional as F
from lavis.datasets.video_utils.event_sampler import (
    sample_equal_sequence,
)
from lavis.datasets.video_utils.randaugment import (
    TemporalConsistentRandomAugment,
    VideoRandomSquareCrop,
)
from lavis.datasets.video_utils.utils import (
    VideoNorm,
    load_video_from_bytes_resample,
    load_video_from_path_decord,
    load_video_from_path_tvio
)

from lavis.datasets.build_tokenizer2 import TokenizerwithIoUtoken
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.omnicaption_dataset import batched_iou


def iou(s1, s2):
    # s1: (2, ) float
    # s2: (2, ) float
    # return: (1, ) float
    s1 = s1.tolist()
    s2 = s2.tolist()

    s1 = [s1[0], s1[1]]
    s2 = [s2[0], s2[1]]

    s1.sort()
    s2.sort()

    if s1[1] <= s2[0] or s2[1] <= s1[0]:
        return 0.0

    intersection = min(s1[1], s2[1]) - max(s1[0], s2[0])
    union = max(s1[1], s2[1]) - min(s1[0], s2[0])

    return intersection / union


def batched_iou(s1, s2):
    # s1: (N, 2)
    # s2: (M, 2)
    # return: (N, M)
    ious = []
    for i in range(s1.shape[0]):
        iou_list = []
        for j in range(s2.shape[0]):
            iou_list.append(iou(s1[i], s2[j]))
        ious.append(iou_list)

    return torch.FloatTensor(np.array(ious))


def remove_duplicate_annotations(timestamps, captions, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []

    for timestamp, caption in zip(timestamps, captions):
        s, e, l = timestamp[0], timestamp[1], caption
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if (
                (abs(s - p_event[0]) <= tol)
                and (abs(e - p_event[1]) <= tol)
                and (l == p_event[2])
            ):
                valid = False
                break
        if valid:
            valid_events.append([s, e, l])

    valid_timestamps = []
    valid_captions = []
    for event in valid_events:
        valid_timestamps.append([event[0], event[1]])
        valid_captions.append(event[2])

    return valid_timestamps, valid_captions


def create_clip_dataset(config):
    dataset_config = config.datasets_cfg

    if dataset_config["tokenize_in_dataloader"]:
        tokenizer = TokenizerwithIoUtoken(dataset_config)
    else:
        tokenizer = None

    train_roots = dataset_config.vis_root_train
    train_anns = dataset_config.ann_paths_train
    assert len(train_roots) == len(
        train_anns
    ), f"The length of train_root and train_anns should be the same, but we got {train_roots} train roots and {len(train_anns)} train anns"

    use_randaug = dataset_config.get("use_randaug", True)
    use_randcrop = dataset_config.get("use_randcrop", True)

    num_augs = dataset_config.get("num_augs", 2)
    aug_level = dataset_config.get("aug_level", 5)

    window_stride = dataset_config.get("window_stride", 1)
    forced_pos_neg_sampling = dataset_config.get(
        "forced_pos_neg_sampling", False
    )
    forced_start_mid_end_sampling = dataset_config.get(
        "forced_start_mid_end_sampling", False
    )
    no_pred_cls = dataset_config.get("no_pred_cls", True)
    p_encourage_pos = dataset_config.get("p_encourage_pos", 0.0)
    fix_fps = dataset_config.get("fix_fps", False)
    num_context_frms = dataset_config.get("num_context_frms", 0)

    train_datasets = []
    for train_root, train_ann in zip(train_roots, train_anns):
        train_set = OmniCaptionDataset(
            vis_root=train_root,
            ann_paths=train_ann,
            fps=dataset_config.fps,
            input_length=dataset_config.input_length,
            image_size=dataset_config.image_size,
            tokenizer=tokenizer,
            use_randcrop=use_randcrop,
            use_randaug=use_randaug,
            num_augs=num_augs,
            aug_level=aug_level,
            forced_pos_neg_sampling=forced_pos_neg_sampling,
            forced_start_mid_end_sampling=forced_start_mid_end_sampling,
            no_pred_cls=no_pred_cls,
            p_encourage_pos=p_encourage_pos,
            fix_fps=fix_fps,
            num_context_frms=num_context_frms,
        )
        train_datasets.append(train_set)

    datasets = {"train": train_datasets}

    if dataset_config.get("vis_root_val", None) is not None:
        val_dataset = EvalOmniCaptionDataset(
            vis_root=dataset_config.vis_root_val[0],
            ann_paths=dataset_config.ann_paths_val[0],
            fps=dataset_config.fps,
            fix_fps=fix_fps,
            input_length=dataset_config.input_length,
            image_size=dataset_config.image_size,
            tokenizer=tokenizer,
            window_stride=window_stride,
            num_context_frms=num_context_frms,
            spatial_crop=dataset_config.get("spatial_crop", None),
            temporal_crop=dataset_config.get("temporal_crop", None)
        )
        datasets["val"] = [val_dataset]

    return datasets


class OmniCaptionDataset(BaseDataset):
    def __init__(
        self,
        vis_root=None,
        ann_paths=[],
        fps=1,
        input_length=100,
        image_size=224,
        tokenizer=None,
        use_randcrop=True,
        use_randaug=True,
        num_augs=2,
        aug_level=5,
        sampling_frame_range=5,
        sampling_interval=1,
        sampling_frame_shuffle=False,
        no_pred_cls=True,
        forced_pos_neg_sampling=False,
        forced_start_mid_end_sampling=False,
        p_encourage_pos=0.0,
        fix_fps=False,
        num_context_frms=0,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_root=vis_root, ann_paths=ann_paths)

        annotation = []
        for ann_path in ann_paths:
            annotation.extend(json.load(open(ann_path, "r")))

        self.annotation = annotation

        self.fps = fps
        self.input_length = input_length
        self.image_size = image_size
        self.use_randcrop = use_randcrop
        self.use_randaug = use_randaug

        self.sampling_frame_range = sampling_frame_range
        self.sampling_interval = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle

        self.forced_pos_neg_sampling = forced_pos_neg_sampling
        self.forced_start_mid_end_sampling = forced_start_mid_end_sampling
        self.no_pred_cls = no_pred_cls
        self.p_encourage_pos = p_encourage_pos
        self.fix_fps = fix_fps
        self.num_context_frms = num_context_frms

        self.video_random_cropper = VideoRandomSquareCrop(self.image_size)
        self.randaug = TemporalConsistentRandomAugment(
            N=num_augs,
            M=aug_level,
            augs=[
                "Identity",
                "Contrast",
                "Brightness",
                "Sharpness",
                "ShearX",
                "ShearY",
                "TranslateX",
                "TranslateY",
                "Rotate",
                "HorizontalFlip",
            ],
        )
        self.norm = VideoNorm()
        self.tokenize = tokenizer

        if os.path.exists(os.path.join(vis_root, "tal_classes.txt")):
            action_dict = {}
            with open(os.path.join(vis_root, "tal_classes.txt"), "r") as f:
                for i, line in enumerate(f):
                    action_dict[line.strip()] = i
            self.action_dict = action_dict
            self.noise_label = len(action_dict)

        elif (
            os.path.exists(os.path.join(vis_root, "kinetics_400_labels.csv"))
            and "gebd" not in ann_paths[0]
        ):
            action_dict = {}
            with open(
                os.path.join(vis_root, "kinetics_400_labels.csv"), "r"
            ) as f:
                for i, line in enumerate(f):
                    action_dict[line.strip().split(",")[1]] = i
            self.action_dict = action_dict
            self.noise_label = len(action_dict)

        else:
            self.action_dict = None
            self.noise_label = 1

    def find_segment_with_largest_iou(
        self, sampling_start, sampling_end, timestamps, captions
    ):
        sampling_segment = [sampling_start, sampling_end]
        iou = batched_iou(
            torch.FloatTensor([sampling_segment]),
            torch.FloatTensor(timestamps),
        )[
            0
        ]  # (1, num_segments)
        max_iou = torch.max(iou)

        if max_iou == 0:
            # this clip does not fall into any segment
            # we assign it to the background class
            matched_caption = "<background>"
            matched_segments = None
            inter_seconds = 0.0
        else:
            max_iou_idx = torch.argmax(iou)
            matched_caption = captions[max_iou_idx]
            matched_segments = timestamps[max_iou_idx]
            inter_seconds = min(sampling_segment[1], matched_segments[1]) - max(
                sampling_segment[0], matched_segments[0]
            )

        return matched_caption, inter_seconds / (
            sampling_segment[1] - sampling_segment[0]
        )

    def sample_pos(self, ann, default_fps, read_frames, pos=None):
        num_segments = len(ann["timestamps"])
        sampling_segment_idx = random.randint(0, num_segments - 1)
        sampling_segment = ann["timestamps"][sampling_segment_idx]

        segment_start_inframe = int(sampling_segment[0] * default_fps)
        segment_end_inframe = int(sampling_segment[1] * default_fps)
        segment_end_inframe = min(segment_end_inframe, read_frames)

        if segment_start_inframe + self.input_length >= read_frames:
            # seg_start --------- seg_start + input_length
            #                | end of vid
            sampling_start_inframe = read_frames - 1 - self.input_length
            sampling_end_inframe = sampling_start_inframe + self.input_length

        elif segment_end_inframe + self.input_length >= read_frames:
            # seg_start -------- seg_end -------- seg_end + input_length
            #                               | end of vid
            sampling_start_inframe = random.randint(
                segment_start_inframe,
                read_frames - 1 - self.input_length,
            )
            sampling_end_inframe = sampling_start_inframe + self.input_length

        else:
            if pos == "start":
                # sampling near the start: seg_start < sampling_start < seg_end
                # theoritically: 0 ~ seg_start - input + 1 (so that end: input ~ seg_start + 1)
                if self.input_length < segment_start_inframe:
                    sampling_start_inframe = random.randint(
                        segment_start_inframe - self.input_length + 1,
                        segment_start_inframe - 1,
                    )  # start - input + 1 ~ start - 1
                    # start + 1 ~ start - 1 + input
                else:
                    sampling_start_inframe = random.randint(
                        0, max(segment_start_inframe - 1, 0)
                    )
                sampling_end_inframe = (
                    sampling_start_inframe + self.input_length
                )

            elif pos == "mid":
                # sampling inside the segment
                if (
                    segment_end_inframe - segment_start_inframe - 1
                    > self.input_length
                ):
                    sampling_start_inframe = random.randint(
                        segment_start_inframe,
                        segment_end_inframe - self.input_length - 1,
                    )
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )

                elif random.random() < 0.5:
                    sampling_start_inframe = segment_start_inframe
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )
                    if sampling_end_inframe >= read_frames:
                        sampling_end_inframe = read_frames - 1
                        sampling_start_inframe = (
                            sampling_end_inframe - self.input_length
                        )

                else:
                    sampling_end_inframe = segment_end_inframe - 1
                    sampling_start_inframe = (
                        sampling_end_inframe - self.input_length
                    )
                    if sampling_start_inframe < 0:
                        sampling_start_inframe = 0
                        sampling_end_inframe = (
                            sampling_start_inframe + self.input_length
                        )

            elif pos == "end":
                if read_frames - self.input_length > segment_end_inframe:
                    sampling_end_inframe = random.randint(
                        segment_end_inframe + 1, read_frames - self.input_length
                    )  # end + 1 ~ vid - input
                    # end + 1 - input ~ vid - input - input

                else:
                    sampling_end_inframe = random.randint(
                        min(segment_end_inframe + 1), read_frames
                    )
                sampling_start_inframe = (
                    sampling_end_inframe - self.input_length
                )

            else:
                sampling_start_inframe = random.randint(
                    segment_start_inframe, segment_end_inframe
                )
                sampling_end_inframe = (
                    sampling_start_inframe + self.input_length
                )

        if sampling_start_inframe < 0 or sampling_end_inframe >= read_frames:
            print(
                pos,
                segment_start_inframe,
                segment_end_inframe,
                sampling_start_inframe,
                sampling_end_inframe,
            )

        return sampling_start_inframe, sampling_end_inframe

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.vis_root, ann["video_name"])
        task = ann["task"].strip()

        if task in [
            "action_recognition",
            "clip_captioning",
            "clip_qa",
            "generic_event_boundary_detection",
        ]:
            if self.use_randcrop:
                if video_path.endswith(".webm"):
                    frames, _ = load_video_from_path_tvio(
                        video_path,
                        frm_sampling_strategy="rand",
                        num_frm=self.input_length,
                        height=int(1.5 * self.image_size),
                        width=int(1.5 * self.image_size),
                    )
                else:
                    frames, _ = load_video_from_path_decord(
                        video_path,
                        frm_sampling_strategy="rand",
                        num_frm=self.input_length,
                        height=int(1.5 * self.image_size),
                        width=int(1.5 * self.image_size),
                    )
            else:
                if video_path.endswith(".webm"):
                    frames, _ = load_video_from_path_tvio(
                        video_path,
                        frm_sampling_strategy="rand",
                        num_frm=self.input_length,
                        height=int(self.image_size),
                        width=int(self.image_size),
                    )
                else:
                    frames, _ = load_video_from_path_decord(
                        video_path,
                        frm_sampling_strategy="rand",
                        num_frm=self.input_length,
                        height=int(self.image_size),
                        width=int(self.image_size),
                    )

            if task == "clip_captioning":
                # assert len(ann["captions"]) == 20
                matched_caption = random.choice(ann["captions"])
            else:
                matched_caption = ann["captions"][0]

            max_iou = None
            sampled_frames = frames
            sampling_start_inseconds, sampling_end_inseconds = None, None

        elif task in ["moment_retrieval"]:
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, ann["video_name"])
            task = ann["task"].strip()

            if self.use_randcrop:
                frames, default_fps = load_video_from_bytes_resample(
                    video_path,
                    height=int(1.5 * self.image_size),
                    width=int(1.5 * self.image_size),
                    fps=-1 if not self.fix_fps else self.fps,
                    return_fps=True,
                )
            else:
                frames, default_fps = load_video_from_bytes_resample(
                    video_path,
                    height=int(self.image_size),
                    width=int(self.image_size),
                    fps=-1 if not self.fix_fps else self.fps,
                    return_fps=True,
                )

            num_frames_read = len(frames)
            pad_left = pad_right = False

            if self.forced_pos_neg_sampling:
                sample_negative = (index % 2 == 1) and (
                    random.random() > self.p_encourage_pos
                )
                sample_positive = not sample_negative

            elif self.forced_start_mid_end_sampling:
                sample_negative = (index % 4 == 0) and (
                    random.random() > self.p_encourage_pos
                )
                sample_positive = not sample_negative

            else:
                sample_positive = sample_negative = True

            if num_frames_read <= self.input_length:
                # we have to pad the video
                pad = np.zeros(
                    (
                        self.input_length - num_frames_read,
                        frames.shape[1],
                        frames.shape[2],
                        frames.shape[3],
                    ),
                    dtype=np.uint8,
                )
                frames = np.concatenate((frames, pad), axis=0)
                sampling_start_inframe = 0
                sampling_end_inframe = self.input_length

            elif (
                self.forced_pos_neg_sampling
                or self.forced_start_mid_end_sampling
            ) and sample_positive:
                # we have to sample a positive segment (that has overlap with one of the ground truth segment)
                if self.forced_pos_neg_sampling:
                    pos = None
                elif index % 4 == 1:
                    pos = "start"
                elif index % 4 == 2:
                    pos = "mid"
                elif index % 4 == 3:
                    pos = "end"
                else:
                    pos = random.choice(["start", "mid", "end"])

                suc = False
                while not suc:
                    (
                        sampling_start_inframe,
                        sampling_end_inframe,
                    ) = self.sample_pos(ann, default_fps, num_frames_read, pos)
                    sampling_start_inframe = max(0, sampling_start_inframe)
                    sampling_end_inframe = min(
                        sampling_end_inframe, num_frames_read - 1
                    )
                    if (
                        sampling_end_inframe - sampling_start_inframe
                        == self.input_length
                    ):
                        suc = True

            elif (
                self.forced_pos_neg_sampling
                or self.forced_start_mid_end_sampling
            ) and sample_negative:
                # we have to sample a negative segment (that has no overlap with any of the ground truth segment)
                # if that is impossible, we just sample a segment that has the smallest overlap with any of the ground truth segment
                starts_inframe = [
                    int(s[0] * default_fps) for s in ann["timestamps"]
                ]
                ends_inframe = [
                    int(s[1] * default_fps) for s in ann["timestamps"]
                ]
                min_start_inframe = min(starts_inframe)
                max_end_inframe = max(ends_inframe)

                if min_start_inframe > self.input_length:
                    # we can sample a segment that has no overlap with any of the ground truth segment
                    sampling_start_inframe = random.randint(
                        0, min_start_inframe - self.input_length
                    )
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )
                elif max_end_inframe + self.input_length < num_frames_read:
                    # we can sample a segment that has no overlap with any of the ground truth segment
                    sampling_start_inframe = random.randint(
                        max_end_inframe, num_frames_read - 1 - self.input_length
                    )
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )
                else:
                    # we randomly sample the very first segment or the very last segment
                    if random.random() < 0.5:
                        sampling_start_inframe = 0
                        sampling_end_inframe = (
                            sampling_start_inframe + self.input_length
                        )
                    else:
                        sampling_end_inframe = num_frames_read - 1
                        sampling_start_inframe = (
                            sampling_end_inframe - self.input_length
                        )

            else:
                # we have sample self.input_length consecutive frames from the video
                sampling_start_inframe = random.randint(
                    0, num_frames_read - 1 - self.input_length
                )
                sampling_end_inframe = (
                    sampling_start_inframe + self.input_length
                )

            sampled_frames = frames[sampling_start_inframe:sampling_end_inframe]

            if sampled_frames.shape[0] < self.input_length:
                if sampling_start_inframe < 0:
                    sampling_start_inframe = 0
                    pad_left = True
                elif sampling_end_inframe >= num_frames_read:
                    sampling_end_inframe = num_frames_read - 1
                    pad_right = True

                assert pad_left or pad_right
                pad = np.zeros(
                    (
                        self.input_length - sampled_frames.shape[0],
                        sampled_frames.shape[1],
                        sampled_frames.shape[2],
                        sampled_frames.shape[3],
                    ),
                    dtype=np.uint8,
                )

                if pad_left:
                    sampled_frames = np.concatenate(
                        (pad, sampled_frames), axis=0
                    )
                else:
                    sampled_frames = np.concatenate(
                        (sampled_frames, pad), axis=0
                    )

            assert (
                sampled_frames.shape[0] == self.input_length
            ), f"we should sample {self.input_length} frames, but we got {sampled_frames.shape[0]} frames"

            sampling_start_inseconds = sampling_start_inframe / default_fps
            sampling_end_inseconds = sampling_end_inframe / default_fps

            if ann["source"] == "qvhighlights":
                captions = ann["captions"] * len(ann["timestamps"])
            else:
                captions = ann["captions"]

            # find the corresponding segments in ann["timestamps"]
            matched_caption, max_iou = self.find_segment_with_largest_iou(
                sampling_start_inseconds,
                sampling_end_inseconds,
                ann["timestamps"],
                captions,
            )

        elif task in ["temporal_action_localization", "dense_video_captioning"]:
            # dense video captioning, temporal action localization
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, ann["video_name"])
            task = ann["task"].strip()

            if isinstance(self.fps, int):
                fps = self.fps
            else:
                assert len(self.fps) == 2
                fps = random.randint(self.fps[0], self.fps[1])

            if self.use_randcrop:
                frames, default_fps = load_video_from_bytes_resample(
                    video_path,
                    height=int(1.5 * self.image_size),
                    width=int(1.5 * self.image_size),
                    fps=-1 if not self.fix_fps else fps,
                    return_fps=True,
                )
            else:
                frames, default_fps = load_video_from_bytes_resample(
                    video_path,
                    height=int(self.image_size),
                    width=int(self.image_size),
                    fps=-1 if not self.fix_fps else fps,
                    return_fps=True,
                )

            num_frames_read = len(frames)
            if self.forced_pos_neg_sampling:
                sample_negative = (index % 2 == 1) and (
                    random.random() > self.p_encourage_pos
                )
                sample_positive = not sample_negative

            elif self.forced_start_mid_end_sampling:
                sample_negative = (index % 4 == 0) and (
                    random.random() > self.p_encourage_pos
                )
                sample_positive = not sample_negative

            else:
                sample_positive = sample_negative = True

            pad_left = pad_right = False
            if num_frames_read <= self.input_length:
                # we have to pad the video
                pad = np.zeros(
                    (
                        self.input_length - num_frames_read,
                        frames.shape[1],
                        frames.shape[2],
                        frames.shape[3],
                    ),
                    dtype=np.uint8,
                )
                frames = np.concatenate((frames, pad), axis=0)
                sampling_start_inframe = 0
                sampling_end_inframe = self.input_length

            elif (
                self.forced_pos_neg_sampling
                or self.forced_start_mid_end_sampling
            ) and sample_positive:
                # we have to sample a positive segment (that has overlap with one of the ground truth segment)
                if self.forced_pos_neg_sampling:
                    pos = None
                elif index % 4 == 1:
                    pos = "start"
                elif index % 4 == 2:
                    pos = "mid"
                elif index % 4 == 3:
                    pos = "end"
                else:
                    pos = random.choice(["start", "mid", "end"])

                suc = False
                while not suc:
                    (
                        sampling_start_inframe,
                        sampling_end_inframe,
                    ) = self.sample_pos(ann, default_fps, num_frames_read, pos)
                    sampling_start_inframe = max(0, sampling_start_inframe)
                    sampling_end_inframe = min(
                        sampling_end_inframe, num_frames_read - 1
                    )
                    if (
                        sampling_end_inframe - sampling_start_inframe
                        == self.input_length
                    ):
                        suc = True

            elif (
                self.forced_pos_neg_sampling
                or self.forced_start_mid_end_sampling
            ) and sample_negative:
                # we have to sample a negative segment (that has no overlap with any of the ground truth segment)
                # if that is impossible, we just sample a segment that has the smallest overlap with any of the ground truth segment
                starts_inframe = [
                    int(s[0] * default_fps) for s in ann["timestamps"]
                ]
                ends_inframe = [
                    int(s[1] * default_fps) for s in ann["timestamps"]
                ]
                min_start_inframe = min(starts_inframe)
                max_end_inframe = max(ends_inframe)

                if min_start_inframe > self.input_length:
                    # we can sample a segment that has no overlap with any of the ground truth segment
                    sampling_start_inframe = random.randint(
                        0, min_start_inframe - self.input_length
                    )
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )
                elif max_end_inframe + self.input_length < num_frames_read:
                    # we can sample a segment that has no overlap with any of the ground truth segment
                    sampling_start_inframe = random.randint(
                        max_end_inframe, num_frames_read - 1 - self.input_length
                    )
                    sampling_end_inframe = (
                        sampling_start_inframe + self.input_length
                    )
                else:
                    # we randomly sample the very first segment or the very last segment
                    if random.random() < 0.5:
                        sampling_start_inframe = 0
                        sampling_end_inframe = (
                            sampling_start_inframe + self.input_length
                        )
                    else:
                        sampling_end_inframe = num_frames_read - 1
                        sampling_start_inframe = (
                            sampling_end_inframe - self.input_length
                        )

            else:
                # we have sample self.input_length consecutive frames from the video
                sampling_start_inframe = random.randint(
                    0, num_frames_read - 1 - self.input_length
                )
                sampling_end_inframe = (
                    sampling_start_inframe + self.input_length
                )

            sampled_frames = frames[sampling_start_inframe:sampling_end_inframe]

            if sampled_frames.shape[0] < self.input_length:
                if sampling_start_inframe < 0:
                    sampling_start_inframe = 0
                    pad_left = True
                elif sampling_end_inframe >= num_frames_read:
                    sampling_end_inframe = num_frames_read - 1
                    pad_right = True

                assert pad_left or pad_right
                pad = np.zeros(
                    (
                        self.input_length - sampled_frames.shape[0],
                        sampled_frames.shape[1],
                        sampled_frames.shape[2],
                        sampled_frames.shape[3],
                    ),
                    dtype=np.uint8,
                )

                if pad_left:
                    sampled_frames = np.concatenate(
                        (pad, sampled_frames), axis=0
                    )
                else:
                    sampled_frames = np.concatenate(
                        (sampled_frames, pad), axis=0
                    )

            assert (
                sampled_frames.shape[0] == self.input_length
            ), f"we should sample {self.input_length} frames, but we got {sampled_frames.shape[0]} frames"

            sampling_start_inseconds = sampling_start_inframe / default_fps
            sampling_end_inseconds = sampling_end_inframe / default_fps

            # find the corresponding segments in ann["timestamps"]
            matched_caption, max_iou = self.find_segment_with_largest_iou(
                sampling_start_inseconds,
                sampling_end_inseconds,
                ann["timestamps"],
                ann["captions"],
            )

        else:
            raise NotImplementedError

        if self.num_context_frms > 0 and (
            not task
            in [
                "action_recognition",
                "clip_captioning",
                "clip_qa",
                "generic_event_boundary_detection",
            ]
        ):
            pad_with_context = True
            if pad_left:
                # pad on the left:
                pre_diff_frms = np.zeros(
                    (
                        self.num_context_frms,
                        sampled_frames.shape[1],
                        sampled_frames.shape[2],
                        sampled_frames.shape[3],
                    ),
                    dtype=np.uint8,
                )
            else:
                pre_context_start = max(
                    sampling_start_inframe - self.num_context_frms - 1, 0
                )
                pre_context_end = max(sampling_start_inframe - 1, 0)
                pre_context_frms = frames[pre_context_start:pre_context_end]
                pre_diff_frms = sampled_frames[0:1] - pre_context_frms

                if pre_diff_frms.shape[0] < self.num_context_frms:
                    pad = np.zeros(
                        (
                            self.num_context_frms - pre_diff_frms.shape[0],
                            sampled_frames.shape[1],
                            sampled_frames.shape[2],
                            sampled_frames.shape[3],
                        ),
                        dtype=np.uint8,
                    )

                    pre_diff_frms = np.concatenate((pad, pre_diff_frms), axis=0)

            if pad_right:
                post_diff_frms = np.zeros(
                    (
                        self.num_context_frms,
                        sampled_frames.shape[1],
                        sampled_frames.shape[2],
                        sampled_frames.shape[3],
                    ),
                    dtype=np.uint8,
                )
            else:
                post_context_start = min(
                    sampling_end_inframe, num_frames_read - 1
                )
                post_context_end = min(
                    post_context_start + self.num_context_frms,
                    num_frames_read - 1,
                )
                post_context_frms = frames[post_context_start:post_context_end]
                post_diff_frms = sampled_frames[-1:] - post_context_frms
                if post_diff_frms.shape[0] < self.num_context_frms:
                    pad = np.zeros(
                        (
                            self.num_context_frms - post_diff_frms.shape[0],
                            sampled_frames.shape[1],
                            sampled_frames.shape[2],
                            sampled_frames.shape[3],
                        ),
                        dtype=np.uint8,
                    )

                    post_diff_frms = np.concatenate(
                        (post_diff_frms, pad), axis=0
                    )

            sampled_frames = np.concatenate(
                [pre_diff_frms, sampled_frames, post_diff_frms], axis=0
            )
        else:
            pad_with_context = False

        vid_frm_array = self.video_random_cropper(sampled_frames)

        if self.use_randaug:
            vid_frm_array = self.randaug(vid_frm_array).permute(0, 3, 1, 2)
        else:
            vid_frm_array = (
                torch.from_numpy(vid_frm_array).float().permute(0, 3, 1, 2)
            )

        video = self.norm(vid_frm_array)
        if pad_with_context:
            pre_frms = video[: self.num_context_frms]
            post_frms = video[-self.num_context_frms :]
            key_frms = video[self.num_context_frms : -self.num_context_frms]
            pre_diff_frms = (
                video[self.num_context_frms : self.num_context_frms + 1]
                - pre_frms
            )
            post_diff_frms = (
                video[-self.num_context_frms - 1 : -self.num_context_frms]
                - post_frms
            )

            video = torch.cat([pre_diff_frms, key_frms, post_diff_frms], dim=0)

        if task == "clip_qa":
            assert (
                len(ann["captions"]) == 1
            ), "Question Answering should have only one sentence"

            caption = matched_caption
            question = caption.split("Question:")[1].split("Answer:")[0]
            answer = caption.split("Answer:")[1].strip()

            text, _ = self.tokenize([answer], None, tokenize_type="caption")
            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize(
                [task + ". " + question], None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "text": text,
                "prompt": prompt,
                "reference_points": -1,
            }

        elif task in ["action_recognition", "clip_captioning"]:
            # clip video captioning, action recognition
            text, _ = self.tokenize(
                [matched_caption], None, tokenize_type="caption"
            )
            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize([task + "."], None, tokenize_type="prompt")[
                0
            ]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "text": text,
                "prompt": prompt,
                "reference_points": -1,
            }

        elif task == "moment_retrieval":
            if matched_caption == "<background>":
                text, _ = self.tokenize(
                    [matched_caption], max_iou, tokenize_type="caption"
                )
            else:
                if self.no_pred_cls:
                    text, _ = self.tokenize(
                        ["something"], max_iou, tokenize_type="caption"
                    )
                else:
                    text, _ = self.tokenize(
                        [matched_caption], max_iou, tokenize_type="caption"
                    )

            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize(
                [task + ". " + ann["captions"][0]],
                None,
                tokenize_type="prompt",
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "text": text,
                "prompt": prompt,
                "reference_points": torch.FloatTensor(
                    [
                        sampling_start_inseconds / ann["duration"],
                        sampling_end_inseconds / ann["duration"],
                    ]
                ),
            }

        else:
            # dense video captioning, temporal action localization
            text, _ = self.tokenize(
                [matched_caption], max_iou, tokenize_type="caption"
            )

            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize([task + "."], None, tokenize_type="prompt")[
                0
            ]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()
            sample = {
                "task": task,
                "video": video,
                "text": text,
                "prompt": prompt,
                "reference_points": torch.FloatTensor(
                    [
                        sampling_start_inseconds / ann["duration"],
                        sampling_end_inseconds / ann["duration"],
                    ]
                ),
            }

        return sample


class EvalOmniCaptionDataset(BaseDataset):
    def __init__(
        self,
        vis_root=None,
        ann_paths=[],
        fps=1,
        input_length=100,
        image_size=224,
        tokenizer=None,
        window_stride=1.0,
        fix_fps=False,
        num_context_frms=0,
        spatial_crop=None,
        temporal_crop=None,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_root=vis_root, ann_paths=ann_paths)
        self.fps = fps
        self.input_length = input_length
        self.image_size = image_size
        self.spatial_crop = spatial_crop
        self.temporal_crop = temporal_crop

        self.norm = VideoNorm()
        self.tokenize = tokenizer
        self.window_stride = window_stride
        self.fix_fps = fix_fps
        self.num_context_frms = num_context_frms

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.vis_root, ann["video_name"])
        task = ann["task"].strip()

        if task in ["action_recognition", "clip_captioning", "clip_qa"]:
            if video_path.endswith(".webm"):
                frames, _, default_fps = load_video_from_path_tvio(
                    video_path,
                    frm_sampling_strategy="uniform",
                    height=self.image_size,
                    width=self.image_size,
                    num_frm=self.input_length,
                    return_fps=True,
                    spatial_crop=self.spatial_crop,
                    temporal_crop=self.temporal_crop,
                )
            else:
                frames, _, default_fps = load_video_from_path_decord(
                    video_path,
                    frm_sampling_strategy="uniform",
                    height=self.image_size,
                    width=self.image_size,
                    num_frm=self.input_length,
                    return_fps=True,
                    spatial_crop=self.spatial_crop,
                    temporal_crop=self.temporal_crop,
                )

            frames = torch.from_numpy(frames)
            vid_frm_array = frames.permute(0, 3, 1, 2).float()
            video = self.norm(vid_frm_array)

            sampled_clips = video
            sampled_timestamps = torch.FloatTensor([[0, 0]])

            if task == "clip_captioning" and "id" in ann:
                video_id = "video" + str(ann["id"])
            elif task == "clip_captioning":
                video_id = os.path.basename(ann["video_name"]).split(".")[0]
            else:
                video_id = 0

        else:
            frames, _, default_fps = load_video_from_path_decord(
                video_path,
                frm_sampling_strategy="all",
                height=self.image_size,
                width=self.image_size,
                fps=-1 if not self.fix_fps else self.fps,
                return_fps=True,
            )

            frames = torch.from_numpy(frames)
            vid_frm_array = frames.permute(0, 3, 1, 2).float()
            video = self.norm(vid_frm_array)
            num_read_frms = video.shape[0]

            if video.shape[0] < self.input_length:
                pad = torch.zeros(
                    (
                        self.input_length - video.shape[0],
                        video.shape[1],
                        video.shape[2],
                        video.shape[3],
                    ),
                    dtype=video.dtype,
                )
                video = torch.cat((video, pad), dim=0)

            # Given len(frames) frames, we need to sample several clips from it
            # and each clip has self.input_length frames,
            # and the stride between two clips is self.window_stride
            num_clips = (
                math.ceil(
                    (video.shape[0] - self.input_length) / self.window_stride
                )
                + 1
            )

            sampled_clips = []
            timestamps = []
            for i in range(num_clips):
                start_inframe = i * self.window_stride
                end_inframe = start_inframe + self.input_length
                sampled_clip = video[start_inframe:end_inframe]
                if sampled_clip.shape[0] < self.input_length:
                    pad = torch.zeros(
                        (
                            self.input_length - sampled_clip.shape[0],
                            sampled_clip.shape[1],
                            sampled_clip.shape[2],
                            sampled_clip.shape[3],
                        )
                    )
                    sampled_clip = torch.cat((sampled_clip, pad), dim=0)

                if self.num_context_frms > 0:
                    pre_start = max(
                        0, start_inframe - self.num_context_frms - 1
                    )
                    pre_end = max(0, start_inframe - 1)
                    pre_clip = video[pre_start:pre_end]
                    if pre_clip.shape[0] < self.num_context_frms:
                        pad_pre = torch.zeros(
                            (
                                self.num_context_frms - pre_clip.shape[0],
                                sampled_clip.shape[1],
                                sampled_clip.shape[2],
                                sampled_clip.shape[3],
                            )
                        )
                        pre_clip = torch.cat((pad_pre, pre_clip), dim=0)

                    pre_diff_clip = sampled_clip[0:1] - pre_clip

                    post_start = min(num_read_frms - 1, end_inframe)
                    post_end = min(
                        num_read_frms - 1, end_inframe + self.num_context_frms
                    )
                    post_clip = video[post_start:post_end]
                    if post_clip.shape[0] < self.num_context_frms:
                        pad_post = torch.zeros(
                            (
                                self.num_context_frms - post_clip.shape[0],
                                sampled_clip.shape[1],
                                sampled_clip.shape[2],
                                sampled_clip.shape[3],
                            )
                        )
                        post_clip = torch.cat((post_clip, pad_post), dim=0)

                    post_diff_clip = sampled_clip[-1:] - post_clip

                    sampled_clip = torch.cat(
                        [pre_diff_clip, sampled_clip, post_diff_clip], dim=0
                    ).unsqueeze(dim=0)

                else:
                    sampled_clip = sampled_clip.unsqueeze(dim=0)

                sampled_clips.append(sampled_clip)

                start_inseconds = start_inframe / default_fps
                end_inseconds = end_inframe / default_fps
                timestamps.append([start_inseconds, end_inseconds])

            sampled_clips = torch.cat(sampled_clips, dim=0)
            sampled_timestamps = torch.FloatTensor(timestamps)

            if "qid" in ann:
                video_id = ann["qid"]
            else:
                video_id = 0

        if task in [
            "action_recognition",
            "clip_captioning",
            "dense_video_caption",
            "temporal_action_localization",
        ]:
            prompt = self.tokenize([task + "."], None, tokenize_type="prompt")[
                0
            ]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()
            raw_prompt = task + "."

        elif task == "moment_retrieval":
            prompt = self.tokenize(
                [task + ". " + ann["captions"][0]],
                None,
                tokenize_type="prompt",
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()
            raw_prompt = task + ". " + ann["captions"][0]

        elif task == "clip_qa":
            caption = ann["captions"][0]
            question = caption.split("Question:")[1].split("Answer:")[0]

            prompt = self.tokenize(
                [task + ". " + question], None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()
            raw_prompt = task + ". " + question

        else:
            raise NotImplementedError

        sample = {
            "video": sampled_clips,
            "text": ann["captions"],
            "timestamps": ann["timestamps"],
            "reference_points": sampled_timestamps,
            "duration": ann["duration"],
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "name": ann["video_name"],
            "task": task,
            "source": ann["source"],
            "video_id": video_id,
        }

        return sample
