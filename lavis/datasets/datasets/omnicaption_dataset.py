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
)
from lavis.datasets.build_tokenizer import TokenizerwithTimetoken
from lavis.datasets.datasets.base_dataset import BaseDataset


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
    """for event in ants:
        s, e, l = event['timestamps'][0], event['timestamps'][1], event['captions']
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)"""

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


def create_dataset(config):
    dataset_config = config.datasets_cfg

    if dataset_config["tokenize_in_dataloader"]:
        tokenizer = TokenizerwithTimetoken(dataset_config)
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

    time_triplet = dataset_config.get("time_triplet", "start_end_cat")
    assert time_triplet in ["start_end_cat", "center_duration_cat"]

    no_pred_cls = dataset_config.get("no_pred_cls", False)
    padding_threshold = dataset_config.get("padding_threshold", 0.0)
    preserve = dataset_config.get("preserve", False)
    safe_range = dataset_config.get("safe_range", 0.0)
    avoid_first_last = dataset_config.get("avoid_first_last", False)
    use_crop1 = dataset_config.get("use_crop1", True)

    max_segments = dataset_config.get("max_segments", -1)
    supervise_with_clipwise_sequence = dataset_config.get(
        "supervise_with_clipwise_sequence", False
    )
    clip_length = dataset_config.get("clip_length", 1)
    mix_clipwise_sequence = dataset_config.get(
        "mix_clipwise_sequence", False
    )
    p_mix = dataset_config.get(
        "p_mix", 0.
    )

    caption_only = dataset_config.get("caption_only", False)
    train_datasets = []
    for train_root, train_ann in zip(train_roots, train_anns):
        train_set = OmniCaptionDataset(
            vis_root=train_root,
            ann_paths=train_ann,
            fps=dataset_config.fps,
            num_frms_per_clip=dataset_config.num_frms_per_clip
            if config.model_cfg.use_video_qformer
            else 1,
            input_length=dataset_config.input_length,
            image_size=dataset_config.image_size,
            random_temporal_crop_proba=dataset_config.random_temporal_crop_proba,
            tokenizer=tokenizer,
            use_randcrop=use_randcrop,
            use_randaug=use_randaug,
            num_augs=num_augs,
            aug_level=aug_level,
            time_triplet=time_triplet,
            no_pred_cls=no_pred_cls,
            padding_threshold=padding_threshold,
            preserve=preserve,
            safe_range=safe_range,
            avoid_first_last=avoid_first_last,
            use_crop1=use_crop1,
            max_segments=max_segments,
            supervise_with_clipwise_sequence=supervise_with_clipwise_sequence,
            clip_length=clip_length,
            mix_clipwise_sequence=mix_clipwise_sequence,
            p_mix=p_mix,
            caption_only=caption_only
        )
        train_datasets.append(train_set)

    datasets = {"train": train_datasets}

    if dataset_config.get("vis_root_val", None) is not None:
        val_dataset = EvalOmniCaptionDataset(
            vis_root=dataset_config.vis_root_val[0],
            ann_paths=dataset_config.ann_paths_val[0],
            fps=dataset_config.fps,
            input_length=dataset_config.input_length,
            image_size=dataset_config.image_size,
            tokenizer=tokenizer,
        )
        datasets["val"] = [val_dataset]

    return datasets


class OmniCaptionDataset(BaseDataset):
    def __init__(
        self,
        vis_root=None,
        ann_paths=[],
        fps=1,
        num_frms_per_clip=1,
        input_length=100,
        image_size=224,
        random_temporal_crop_proba=0.0,
        tokenizer=None,
        use_randcrop=True,
        use_randaug=True,
        num_augs=2,
        aug_level=5,
        sampling_frame_range=5,
        sampling_interval=1,
        sampling_frame_shuffle=False,
        no_pred_cls=False,
        time_triplet="start_end_cat",
        padding_threshold=0,
        preserve=False,
        safe_range=0.0,
        avoid_first_last=False,
        use_crop1=True,
        max_segments=-1,
        supervise_with_clipwise_sequence=False,
        clip_length=1,
        mix_clipwise_sequence=False,
        p_mix=0.,
        caption_only=False,
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
        self.num_frms_per_clip = num_frms_per_clip
        self.input_length = input_length
        self.image_size = image_size
        self.random_temporal_crop_proba = random_temporal_crop_proba
        self.use_randcrop = use_randcrop
        self.use_randaug = use_randaug

        self.sampling_frame_range = sampling_frame_range
        self.sampling_interval = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle

        self.no_pred_cls = no_pred_cls
        self.time_triplet = time_triplet
        self.padding_threshold = padding_threshold
        self.preserve = preserve
        self.safe_range = safe_range
        self.avoid_first_last = avoid_first_last
        self.use_crop1 = use_crop1
        self.max_segments = max_segments

        self.supervise_with_clipwise_sequence = supervise_with_clipwise_sequence
        self.clip_length = clip_length

        self.mix_clipwise_sequence = mix_clipwise_sequence
        self.p_mix = p_mix
        self.caption_only = caption_only

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

    def repeat_annotations(self, anns):
        updated_anns = anns.copy()
        captions = anns["captions"]
        timestamps = anns["timestamps"]
        duration = anns["duration"]
        updated_captions = []
        updated_timestamps = []
        repeat_times = self.max_segments // len(timestamps)

        for caption, timestamp in zip(captions, timestamps):
            updated_captions.extend([caption] * repeat_times)
            updated_timestamps.extend([timestamp] * repeat_times)

        updated_anns["captions"] = updated_captions
        updated_anns["timestamps"] = updated_timestamps
        return updated_anns

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

    def prepare_clipwise_sequence(self, duration, timestamps, sentences):
        clip_timestamps = []
        clip_captions = []
        clip_ious = []

        sampled_length = int(duration * self.fps)
        num_clips = sampled_length // self.clip_length + 1

        for i in range(num_clips):
            clip_start = i * self.clip_length
            clip_end = (i + 1) * self.clip_length

            clip_start_inseconds = clip_start / self.fps
            clip_end_inseconds = min(clip_end / self.fps, duration)

            if clip_end_inseconds - clip_start_inseconds < 1e-4:
                continue

            clip_timestamps.append([clip_start_inseconds, clip_end_inseconds])
            matched_caption, matched_iou = self.find_segment_with_largest_iou(
                clip_start_inseconds, clip_end_inseconds, timestamps, sentences
            )
            clip_captions.append(matched_caption)
            clip_ious.append(matched_iou)

        return clip_timestamps, clip_captions, clip_ious

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.vis_root, ann["video_name"])
        task = ann["task"]

        if task in [
            "action_recognition",
            "clip_captioning",
            "clip_qa",
            "generic_event_boundary_detection",
        ]:
            if self.use_randcrop:
                frames, frame_masks = load_video_from_path_decord(
                    video_path,
                    frm_sampling_strategy="rand",
                    num_frm=self.input_length,
                    height=int(1.5 * self.image_size),
                    width=int(1.5 * self.image_size),
                )
            else:
                frames, frame_masks = load_video_from_path_decord(
                    video_path,
                    frm_sampling_strategy="rand",
                    num_frm=self.input_length,
                    height=int(self.image_size),
                    width=int(self.image_size),
                )
            updated_ann = ann

        elif task in ["moment_retrieval"]:
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, ann["video_name"])
            task = ann["task"]

            if self.use_randcrop:
                frames = load_video_from_bytes_resample(
                    video_path,
                    height=int(1.5 * self.image_size),
                    width=int(1.5 * self.image_size),
                    fps=self.fps,
                )
            else:
                frames = load_video_from_bytes_resample(
                    video_path,
                    height=int(self.image_size),
                    width=int(self.image_size),
                    fps=self.fps,
                )

            ann["sequence"] = frames
            # calculate the binary saliency label according to the timestamps
            timestamps = ann["timestamps"]
            # duration = ann["duration"]
            num_frames = len(frames)
            saliency_label = np.zeros(num_frames)
            for timestamp in timestamps:
                start = int(timestamp[0] * self.fps)
                end = int(timestamp[1] * self.fps)
                saliency_label[start:end] = 1

            ann["saliency_label"] = saliency_label

            updated_ann = sample_equal_sequence(
                batch=ann,
                num_steps=self.input_length,
                is_training=True,
                p=self.random_temporal_crop_proba,
                fps=self.fps,
                preserve=self.preserve,
                threshold=self.padding_threshold,
                safe_range=self.safe_range,
                avoid_first_last=self.avoid_first_last,
                use_crop1=self.use_crop1,
                num_bins=self.tokenize.num_bins,
            )

            frames = updated_ann["sequence"]
            frame_masks = updated_ann["sequence_mask"]
            saliency_label = updated_ann["saliency_label"]

        elif task in ["temporal_action_localization", "dense_video_captioning"]:
            # dense video captioning, temporal action localization
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, ann["video_name"])
            task = ann["task"]

            if self.use_randcrop:
                frames = load_video_from_bytes_resample(
                    video_path,
                    height=int(1.5 * self.image_size),
                    width=int(1.5 * self.image_size),
                    fps=self.fps,
                )
            else:
                frames = load_video_from_bytes_resample(
                    video_path,
                    height=int(self.image_size),
                    width=int(self.image_size),
                    fps=self.fps,
                )

            ann["sequence"] = frames

            updated_ann = sample_equal_sequence(
                batch=ann,
                num_steps=self.input_length,
                is_training=True,
                p=self.random_temporal_crop_proba,
                fps=self.fps,
                preserve=self.preserve,
                threshold=self.padding_threshold,
                safe_range=self.safe_range,
                avoid_first_last=self.avoid_first_last,
                use_crop1=self.use_crop1,
                num_bins=self.tokenize.num_bins,
            )

            frames = updated_ann["sequence"]
            frame_masks = updated_ann["sequence_mask"]

        else:
            raise NotImplementedError

        vid_frm_array = self.video_random_cropper(frames)

        if self.use_randaug:
            vid_frm_array = self.randaug(vid_frm_array).permute(0, 3, 1, 2)
        else:
            # print("Not using randaug")
            vid_frm_array = (
                torch.from_numpy(vid_frm_array).float().permute(0, 3, 1, 2)
            )

        video = self.norm(vid_frm_array)
        video_mask = torch.FloatTensor(frame_masks).long()

        if task == "moment_retrieval":
            assert (
                len(updated_ann["captions"]) == 1 or self.repeat_times > 0
            ), "Moment retrieval should have only one sentence"

            target_segments_for_tokenize = updated_ann["timestamps"]
            if self.supervise_with_clipwise_sequence:
                (
                    clip_timestamps,
                    clip_captions,
                    clip_ious,
                ) = self.prepare_clipwise_sequence(
                    updated_ann["duration"],
                    updated_ann["timestamps"],
                    updated_ann["captions"],
                )
                # create the target for decoder
                text, _, _ = self.tokenize(
                    clip_captions,
                    clip_timestamps,
                    updated_ann["duration"],
                    segment_masks=None,
                    is_gebd=(task == "generic_event_boundary_detection"),
                    ious=clip_ious,
                )
                text["input_ids"].squeeze_()
                text["attention_mask"].squeeze_()

            else:
                target_captions_for_tokenize = updated_ann["captions"]
                if self.no_pred_cls:
                    target_captions_for_tokenize = ["something"] * len(updated_ann["captions"])

                if self.time_triplet == "start_end_cat":
                    target_segments_for_tokenize = target_segments_for_tokenize
                elif self.time_triplet == "center_duration_cat":
                    new_target_segments_for_tokenize = []
                    for seg in target_segments_for_tokenize:
                        new_target_segments_for_tokenize.append(
                            (
                                (seg[0] + seg[1]) / 2,
                                seg[1] - seg[0],
                            )
                        )
                    target_segments_for_tokenize = (
                        new_target_segments_for_tokenize
                    )
                else:
                    raise NotImplementedError

                # create the target for decoder
                text, _, _ = self.tokenize(
                    target_captions_for_tokenize,
                    target_segments_for_tokenize,
                    updated_ann["duration"],
                    segment_masks=None,
                )
                text["input_ids"].squeeze_()
                text["attention_mask"].squeeze_()

            prompt_txt = updated_ann["captions"][0]
            prompt = self.tokenize(
                [
                    task
                    + ". Retrieve the moment from the following video: "
                    + prompt_txt
                ],
                None,
                None,
                tokenize_type="prompt",
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "video_mask": video_mask,
                "text": text,
                "prompt": prompt,
                "class_id": ann["class_id"] if "class_id" in ann else -1,
            }

        elif task == "clip_qa":
            assert (
                len(updated_ann["captions"]) == 1
            ), "Moment retrieval should have only one sentence"

            caption = updated_ann["captions"][0]
            question = caption.split("Question:")[1].split("Answer:")[0]
            answer = caption.split("Answer:")[1].strip()

            text, _, _ = self.tokenize(
                [answer],
                updated_ann["timestamps"],
                updated_ann["duration"],
            )
            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize(
                [task + ". " + question], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "video_mask": video_mask,
                "text": text,
                "text2": -1,
                "prompt": prompt,
                "class_id": ann["class_id"] if "class_id" in ann else -1,
            }

        elif task in ["action_recognition", "clip_captioning"]:
            # clip video captioning, action recognition
            text, _, _ = self.tokenize(
                updated_ann["captions"],
                updated_ann["timestamps"],
                updated_ann["duration"],
            )
            text["input_ids"].squeeze_()
            text["attention_mask"].squeeze_()

            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "video_mask": video_mask,
                "text": text,
                "text2": -1,
                "prompt": prompt,
                "class_id": ann["class_id"] if "class_id" in ann else -1,
            }

        else:
            # dense video captioning, temporal action localization
            target_segments_for_tokenize = updated_ann["timestamps"]
            if self.supervise_with_clipwise_sequence:
                (
                    clip_timestamps,
                    clip_captions,
                    clip_ious,
                ) = self.prepare_clipwise_sequence(
                    updated_ann["duration"],
                    updated_ann["timestamps"],
                    updated_ann["captions"],
                )
                # create the target for decoder
                text, _, _ = self.tokenize(
                    clip_captions,
                    clip_timestamps,
                    updated_ann["duration"],
                    segment_masks=None,
                    is_gebd=(task == "generic_event_boundary_detection"),
                    ious=clip_ious,
                )
                text["input_ids"].squeeze_()
                text["attention_mask"].squeeze_()

            else:
                target_captions_for_tokenize = updated_ann["captions"]
                if self.no_pred_cls:
                    new_target_captions_for_tokenize = []
                    for cap in target_captions_for_tokenize:
                        if cap != "<background>":
                            new_target_captions_for_tokenize.append("something")
                        else:
                            new_target_captions_for_tokenize.append(cap)
                    target_captions_for_tokenize = (
                        new_target_captions_for_tokenize
                    )

                if self.time_triplet == "start_end_cat":
                    target_segments_for_tokenize = target_segments_for_tokenize
                elif self.time_triplet == "center_duration_cat":
                    new_target_segments_for_tokenize = []
                    for seg in target_segments_for_tokenize:
                        new_target_segments_for_tokenize.append(
                            (
                                (seg[0] + seg[1]) / 2,
                                seg[1] - seg[0],
                            )
                        )
                    target_segments_for_tokenize = (
                        new_target_segments_for_tokenize
                    )
                else:
                    raise NotImplementedError


                if self.mix_clipwise_sequence:
                    (
                        clip_timestamps,
                        clip_captions,
                        clip_ious,
                    ) = self.prepare_clipwise_sequence(
                        updated_ann["duration"],
                        updated_ann["timestamps"],
                        updated_ann["captions"],
                    )
                    
                    if random.random() > self.p_mix:
                        ious = [1.0] * len(target_captions_for_tokenize)
                        target_captions_for_tokenize.extend(clip_captions)
                        target_segments_for_tokenize.extend(clip_timestamps)
                        ious.extend(clip_ious)
                    else:
                        ious = None
                
                else:
                    ious = None
                
                if self.caption_only:
                    # for pretraining
                    captions = ""
                    for c in target_captions_for_tokenize:
                        captions += (c.strip() + " ")
                    
                    text, _, _ = self.tokenize(
                        [captions],
                        [[-1.0, -1.0]],
                        None,
                        segment_masks=None,
                        is_gebd=(task == "generic_event_boundary_detection"),
                        ious=ious
                    )
                else:
                    # create the target for decoder
                    text, _, _ = self.tokenize(
                        target_captions_for_tokenize,
                        target_segments_for_tokenize,
                        updated_ann["duration"],
                        segment_masks=None,
                        is_gebd=(task == "generic_event_boundary_detection"),
                        ious=ious
                    )
                text["input_ids"].squeeze_()
                text["attention_mask"].squeeze_()

            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "task": task,
                "video": video,
                "video_mask": video_mask,
                "text": text,
                "prompt": prompt,
                "class_id": ann["class_id"] if "class_id" in ann else -1,
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
        # self.random_temporal_crop_proba = random_temporal_crop_proba

        self.norm = VideoNorm()
        self.tokenize = tokenizer

    def collater(self, samples):
        # samples:
        #   "video": torch.Tensor [T 3 H W]
        #   "text": List[str],
        #   "timestamps": List[List[float]],
        #   "duration": float,
        #   "prompt":
        #       "input_ids",
        #       "attention_mask".

        (
            video_list,
            video_mask_list,
            text_list,
            timestamps_list,
            duration_list,
            prompt_id_list,
            prompt_attn_list,
            raw_prompt_list,
            video_id_list,
            name_list,
            task_list,
            source_list,
            class_id_list,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
        for sample in samples:
            video_list.append(sample["video"])
            video_mask_list.append(sample["video_mask"])
            text_list.append(sample["text"])
            timestamps_list.append(sample["timestamps"])
            duration_list.append(sample["duration"])
            prompt_id_list.append(sample["prompt"]["input_ids"])
            prompt_attn_list.append(sample["prompt"]["attention_mask"])
            raw_prompt_list.append(sample["raw_prompt"])
            video_id_list.append(sample.get("video_id", None))

            name_list.append(sample["name"])
            task_list.append(sample["task"])
            source_list.append(sample["source"])
            class_id_list.append(
                sample["class_id"] if "class_id" in sample else -1
            )

        prompt_list = {
            "input_ids": torch.stack(prompt_id_list, dim=0),
            "attention_mask": torch.stack(prompt_attn_list, dim=0),
        }

        class_id_list = torch.tensor(class_id_list)

        output = {
            "video_id": video_id_list,
            "video": torch.stack(video_list, dim=0),
            "video_mask": torch.stack(video_mask_list, dim=0),
            "text": text_list,
            "timestamps": timestamps_list,
            "duration": duration_list,
            "prompt": prompt_list,
            "raw_prompt": raw_prompt_list,
            "name": name_list,
            "task": task_list,
            "source": source_list,
            "class_id": class_id_list,
        }
        return output

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.vis_root, ann["video_name"])
        task = ann["task"]

        if task in ["action_recognition", "clip_captioning", "clip_qa"]:
            frames, frame_masks = load_video_from_path_decord(
                video_path,
                frm_sampling_strategy="uniform",
                num_frm=self.input_length,
                height=self.image_size,
                width=self.image_size,
                # fps=self.fps,
            )

        else:
            try:
                frames, frame_masks = load_video_from_path_decord(
                    video_path,
                    frm_sampling_strategy="uniform",
                    num_frm=self.input_length,
                    height=self.image_size,
                    width=self.image_size,
                    fps=self.fps,
                )
            except:
                index += 1
                ann = self.annotation[index]
                video_path = os.path.join(self.vis_root, ann["video_name"])
                frames, frame_masks = load_video_from_path_decord(
                    video_path,
                    frm_sampling_strategy="uniform",
                    num_frm=self.input_length,
                    height=self.image_size,
                    width=self.image_size,
                    fps=self.fps,
                )

        frames = torch.from_numpy(frames)
        vid_frm_array = frames.permute(0, 3, 1, 2).float()
        video = self.norm(vid_frm_array)
        video_mask = torch.FloatTensor(frame_masks).long()

        sentences = ann["captions"]
        timestamps = ann["timestamps"]
        duration = ann["duration"]

        # during inference
        if task == "dense_video_captioning":
            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task,
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
            }

        elif task == "moment_retrieval":
            prompt_txt = sentences[0]
            prompt = self.tokenize(
                [
                    task
                    + ". Retrieve the moment from the following video: "
                    + prompt_txt
                ],
                None,
                None,
                tokenize_type="prompt",
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            if "qid" not in ann:
                ann["qid"] = -1

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task + ". " + sentences[0],
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
                "video_id": ann["qid"],
            }

        elif task == "action_recognition":
            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task,
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
                "class_id": ann["class_id"],
            }

        elif task == "clip_captioning":
            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()
            video_id = ann["id"]

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task,
                "video_id": "video" + str(video_id),
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
            }

        elif task == "clip_qa":
            assert len(sentences) == 1, "Clip QA should have only one sentence"
            question = sentences[0].split("Question:")[1].split("Answer:")[0]
            answer = sentences[0].split("Answer:")[1].strip()

            prompt = self.tokenize(
                [task + ". " + question], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": [answer],
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task + ". " + question,
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
            }

        elif task == "temporal_action_localization":
            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task,
                "name": ann["video_name"],
                "task": ann["task"],
                "source": ann["source"],
            }

        elif task == "generic_event_boundary_detection":
            prompt = self.tokenize(
                [task + " ."], None, None, tokenize_type="prompt"
            )[0]
            prompt["input_ids"].squeeze_()
            prompt["attention_mask"].squeeze_()

            sample = {
                "video": video,
                "video_mask": video_mask,
                "text": sentences,
                "timestamps": timestamps,
                "duration": duration,
                "prompt": prompt,
                "raw_prompt": task,
                "name": ann["video_key"],
                "task": ann["task"],
                "source": ann["source"],
            }

        else:
            raise NotImplementedError

        return sample
