"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
import json
import numpy as np

import torch
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import Boxes, ImageList

from lavis.datasets.build_tokenizer import (
    TokenizerwithBoxtoken,
)
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.detection_augmentation import build_augmentation
from od_util.box_ops import visualize_boxes


def generate_mask_for_padded_images(padded_images, image_sizes):
    # case1: [B, T, 3, H, W], image_sizes: [B, T, 2]
    # case2: [B, 3, H, W], image_sizes: [B, 2]
    # print(padded_images.shape, image_sizes.shape)
    is_template = True
    if padded_images.dim() == 5:
        true_B, T, _, H, W = padded_images.shape
        padded_images = padded_images.view(true_B * T, -1, H, W)
        image_sizes = image_sizes.view(true_B * T, -1)
        is_template = False

    B, _, H, W = padded_images.shape
    mask = torch.ones((B, H, W), dtype=torch.bool)
    for m, ori_img_size in zip(mask, image_sizes):
        m[: int(ori_img_size[0]), : int(ori_img_size[1])] = False

    if not is_template:
        mask = mask.view(true_B, T, H, W)

    return mask


def isvalid(bbox, min_len=5, min_ratio=0.01):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return False

    if h < min_len or w < min_len:
        return False

    if h / w < min_ratio or w / h < min_ratio:
        return False

    return True


def create_sot_dataset(config):
    dataset_config = config.datasets_cfg

    if dataset_config["tokenize_in_dataloader"]:
        tokenizer = TokenizerwithBoxtoken(dataset_config)
    else:
        tokenizer = None

    train_roots = dataset_config.vis_root_train
    train_anns = dataset_config.ann_paths_train
    assert len(train_roots) == len(
        train_anns
    ), f"The length of train_root and train_anns should be the same, but we got {train_roots} train roots and {len(train_anns)} train anns"
    repeat_times = dataset_config.get("repeat_times", 1)
    repeat_perturbation = dataset_config.get("repeat_perturbation", 0.0)
    use_extracted_frames = dataset_config.get("use_extracted_frames", False)

    crop_enabled = dataset_config.get("crop_enabled", True)
    crop_type = dataset_config.get("crop_type", "absolute")
    crop_size = dataset_config.get("crop_size", (384, 600))
    augmentations = dataset_config.get("augmentations", [])

    sampling_frame_range = dataset_config.get("sampling_frame_range", 200)
    sampling_interval = dataset_config.get("sampling_interval", 1)

    min_size_train = dataset_config.get(
        "min_size_train",
        [490, 518, 546, 574, 602, 630, 658, 686, 714, 742, 770],
    )
    max_size_train = dataset_config.get("max_size_train", 1333)
    min_size_test = dataset_config.get("min_size_test", 480)
    max_size_test = dataset_config.get("max_size_test", 1333)
    random_flip = dataset_config.get("random_flip", "flip_by_clip")

    use_randaug = dataset_config.get("use_randaug", True)
    box_in_prompt = dataset_config.get("box_in_prompt", True)
    language_in_prompt = dataset_config.get("language_in_prompt", False)
    score_threshold = dataset_config.get("score_threshold", 0.0)
    p_use_language = dataset_config.get("p_use_language", 0.0)

    train_datasets = []
    for train_root, train_ann in zip(train_roots, train_anns):
        train_set = OmniCaptionSOTDataset(
            vis_root=train_root,
            ann_paths=train_ann,
            fps=dataset_config.fps,
            input_length=dataset_config.input_length,
            sampling_frame_range=sampling_frame_range,
            sampling_interval=sampling_interval,
            image_size=dataset_config.image_size,
            random_temporal_crop_proba=dataset_config.random_temporal_crop_proba,
            tokenizer=tokenizer,
            use_randaug=use_randaug,
            with_sen_mr=dataset_config.with_sen_mr,
            repeat_times=repeat_times,
            repeat_perturbation=repeat_perturbation,
            use_extracted_frames=use_extracted_frames,
            crop_enabled=crop_enabled,
            crop_type=crop_type,
            crop_size=crop_size,
            min_size_train=min_size_train,
            max_size_train=max_size_train,
            min_size_test=min_size_test,
            max_size_test=max_size_test,
            random_flip=random_flip,
            augmentations=augmentations,
            box_in_prompt=box_in_prompt,
            language_in_prompt=language_in_prompt,
            score_threshold=score_threshold,
            p_use_language=p_use_language,
            is_train=True,
        )
        train_datasets.append(train_set)

    datasets = {"train": train_datasets}

    if dataset_config.get("vis_root_val", None) is not None:
        val_dataset = OmniCaptionSOTDataset(
            vis_root=dataset_config.vis_root_val[0],
            ann_paths=dataset_config.ann_paths_val[0],
            fps=dataset_config.fps,
            input_length=dataset_config.input_length,
            image_size=dataset_config.image_size,
            random_temporal_crop_proba=dataset_config.random_temporal_crop_proba,
            tokenizer=tokenizer,
            use_randaug=use_randaug,
            with_sen_mr=dataset_config.with_sen_mr,
            repeat_times=repeat_times,
            repeat_perturbation=repeat_perturbation,
            use_extracted_frames=use_extracted_frames,
            crop_enabled=crop_enabled,
            crop_type=crop_type,
            crop_size=crop_size,
            min_size_train=min_size_train,
            max_size_train=max_size_train,
            min_size_test=min_size_test,
            max_size_test=max_size_test,
            random_flip=random_flip,
            augmentations=augmentations,
            box_in_prompt=box_in_prompt,
            language_in_prompt=language_in_prompt,
            score_threshold=score_threshold,
            p_use_language=0.0,
            is_train=False,
        )
        datasets["val"] = [val_dataset]

    return datasets


class OmniCaptionSOTDataset(BaseDataset):
    def __init__(
        self,
        vis_root=None,
        ann_paths=[],
        fps=1,
        input_length=100,
        sampling_frame_range=200,
        sampling_interval=1,
        image_size=224,
        random_temporal_crop_proba=0.0,
        tokenizer=None,
        use_randaug=True,
        with_sen_mr=False,
        repeat_times=1,
        repeat_perturbation=0.0,
        use_extracted_frames=False,
        crop_enabled=True,
        crop_type="absolute",
        crop_size=(384, 600),
        min_size_train=[490, 518, 546, 574, 602, 630, 658, 686, 714, 742, 770],
        max_size_train=1333,
        min_size_test=480,
        max_size_test=1333,
        random_flip="flip_by_clip",
        augmentations=[],
        box_in_prompt=True,
        language_in_prompt=False,
        score_threshold=0.0,
        p_use_language=0.0,
        is_train=True,
        filter_invalid=True,
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
        if is_train and filter_invalid:
            filtered_annotation = []
            for ann in annotation:
                filtered_ann = ann.copy()
                assert (
                    len(ann["bboxes"])
                    == len(ann["file_names"])
                    == len(ann["areas"])
                )
                filtered_bboxes = []
                filtered_file_names = []
                filtered_areas = []
                for bbox, area, file_name in zip(
                    ann["bboxes"], ann["areas"], ann["file_names"]
                ):
                    if not isvalid(bbox):
                        continue
                    filtered_bboxes.append(bbox)
                    filtered_areas.append(area)
                    filtered_file_names.append(file_name)

                if len(filtered_bboxes) == 0:
                    continue

                filtered_ann["bboxes"] = filtered_bboxes
                filtered_ann["areas"] = filtered_areas
                filtered_ann["file_names"] = filtered_file_names
                filtered_ann["length"] = len(filtered_bboxes)
                filtered_annotation.append(filtered_ann)

            self.annotation = filtered_annotation

        if not is_train and "train" in ann_paths[0]:
            self.annotation = self.annotation[:10]

        """elif not is_train:
            trackingnet_res = json.load(open("./trackingnet_test.json"))[
                "Per_vid_re"
            ]
            trackingnet_res = dict(
                sorted(trackingnet_res.items(), key=lambda x: x[1][0])
            )
            challenging_videos = list(trackingnet_res.keys())[:100]
            challenging_videos = [v.split(".")[0] for v in challenging_videos]
            self.annotation = [
                ann
                for ann in self.annotation
                if os.path.basename(ann["video_name"]) in challenging_videos
            ]"""

        self.fps = fps
        self.input_length = input_length
        self.image_size = image_size
        self.random_temporal_crop_proba = random_temporal_crop_proba
        self.use_randaug = use_randaug
        self.with_sen_mr = with_sen_mr
        self.repeat_times = repeat_times
        self.repeat_perturbation = repeat_perturbation
        self.use_extracted_frames = use_extracted_frames

        self.sampling_frame_range = sampling_frame_range
        self.sampling_interval = sampling_interval

        # self.sampling_frame_shuffle = sampling_frame_shuffle
        self.crop_enabled = crop_enabled
        self.is_train = is_train

        self.box_in_prompt = box_in_prompt
        self.language_in_prompt = language_in_prompt
        self.score_threshold = score_threshold
        self.p_use_language = p_use_language

        PIXEL_MEAN = [123.675, 116.280, 103.530]
        PIXEL_STD = [58.395, 57.120, 57.375]
        pixel_mean = torch.Tensor(PIXEL_MEAN).view(3, 1, 1)
        pixel_std = torch.Tensor(PIXEL_STD).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        if crop_enabled and is_train:
            augs_nocrop, augs = build_augmentation(
                is_train,
                sampling_frame_num=input_length,
                crop_enabled=True,
                crop_type=crop_type,
                crop_size=crop_size,
                min_size_train=min_size_train,
                max_size_train=max_size_train,
                min_size_test=min_size_test,
                max_size_test=max_size_test,
                random_flip=random_flip,
                augmentations=augmentations,
            )
        else:
            augs = build_augmentation(
                is_train,
                sampling_frame_num=input_length,
                crop_enabled=False,
                crop_type=crop_type,
                crop_size=crop_size,
                min_size_train=min_size_train,
                max_size_train=max_size_train,
                min_size_test=min_size_test,
                max_size_test=max_size_test,
                random_flip=random_flip,
                augmentations=augmentations,
            )
            augs_nocrop = None

        self.augmentations = T.AugmentationList(augs)
        if augs_nocrop is not None:
            self.augmentations_nocrop = T.AugmentationList(augs_nocrop)
        else:
            self.augmentations_nocrop = None

        self.tokenize = tokenizer

    def get_template(self, img, bbox):
        # img: [3, H, W]
        # bbox: [4]
        # return: [3, H, W]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_crop = img[:, y1:y2, x1:x2]
        # print(img.shape, bbox, img_crop.shape)
        return img_crop

    def collater(self, batch):
        (
            task_list,
            video_list,
            prompt_box_list,
            text_id_list,
            text_attn_list,
            gt_boxes_list,
            prompt_id_list,
            prompt_attn_list,
            blip2query_list,
            template_query_score_list,
            image_sizes_list,
            height_list,
            width_list,
            file_names_list,
            length_list,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        B = len(batch)
        for sample in batch:
            task_list.append(sample["task"])
            # T x 3 x H x W
            video_frms = sample["video"]
            for frm in video_frms:
                video_list.append(frm)

            prompt_box_list.append(sample["prompt_box"])
            text_id_list.append(sample["text"]["input_ids"])
            text_attn_list.append(sample["text"]["attention_mask"])
            gt_boxes_list.append(sample["gt_boxes"].squeeze(0))

            prompt_id_list.append(sample["prompt"]["input_ids"])
            prompt_attn_list.append(sample["prompt"]["attention_mask"])
            blip2query_list.append(sample["blip2query"])
            template_query_score_list.append(sample["template_query_score"])

            image_sizes_list.append(sample["image_sizes"])
            height_list.append(sample["height"])
            width_list.append(sample["width"])
            file_names_list.append(sample["file_names"])
            length_list.append(sample["length"])

        video_list = ImageList.from_tensors(video_list, size_divisibility=14)
        prompt_box_list = torch.stack(prompt_box_list, dim=0)
        image_sizes_list = torch.stack(image_sizes_list, dim=0)

        video = video_list.tensor.view(
            B,
            -1,
            3,
            video_list.tensor.shape[-2],
            video_list.tensor.shape[-1],
        )
        image_mask = generate_mask_for_padded_images(video, image_sizes_list)

        text_list = {
            "input_ids": torch.stack(text_id_list, dim=0),
            "attention_mask": torch.stack(text_attn_list, dim=0),
        }
        gt_boxes_list = torch.stack(gt_boxes_list, dim=0)
        prompt_list = {
            "input_ids": torch.stack(prompt_id_list, dim=0),
            "attention_mask": torch.stack(prompt_attn_list, dim=0),
        }

        batch = {
            "task": task_list,
            "video": video,
            "prompt_box": prompt_box_list,
            "text": text_list,
            "gt_boxes": gt_boxes_list,
            "prompt": prompt_list,
            "blip2query": blip2query_list,
            "template_query_score": template_query_score_list,
            "image_mask": image_mask,
            "image_sizes": image_sizes_list,
            "height": height_list,
            "width": width_list,
            "file_names": file_names_list,
            "length": length_list,
        }

        return batch

    def __getitem__(self, index):
        ann = self.annotation[index]
        # video_path = os.path.join(self.vis_root, ann["video_name"])
        task = ann["task"]

        # single object tracking
        ann = self.annotation[index]

        file_names = ann["file_names"]
        height = ann["height"]
        width = ann["width"]
        video_length = ann["length"]
        bboxes = ann["bboxes"]

        if self.is_train:
            ref_frame = random.randrange(video_length)
            start_idx = max(0, ref_frame - self.sampling_frame_range)
            start_interval = max(0, ref_frame - self.sampling_interval + 1)
            end_idx = min(
                video_length, ref_frame + self.sampling_frame_range + 1
            )
            end_interval = min(video_length, ref_frame + self.sampling_interval)

            selected_idx = np.random.choice(
                np.array(
                    list(range(start_idx, start_interval))
                    + list(range(end_interval, end_idx))
                ),
                self.input_length - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            ref_frame = selected_idx[0]

        else:
            selected_idx = range(video_length)  # [: self.input_length]

        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations

        # template = None
        template_box = None
        images = []
        boxes = []
        aug_shapes = []
        selected_file_names = []
        for frame_idx in selected_idx:
            frame_path = os.path.join(self.vis_root, file_names[frame_idx])
            image = utils.read_image(frame_path, format="RGB")
            selected_file_names.append(file_names[frame_idx])

            aug_input = T.AugInput(image)
            transforms = selected_augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]

            aug_image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
            images.append(self.normalizer(aug_image))

            if not self.is_train and frame_idx == 0:
                bbox_ann = {
                    "bbox": [bboxes[frame_idx]],
                    "bbox_mode": utils.BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                aug_bbox = utils.transform_instance_annotations(
                    bbox_ann, transforms, image_shape
                )["bbox"]
                aug_bbox = torch.FloatTensor(aug_bbox)  # .unsqueeze(0)
                aug_bbox_raw = aug_bbox.clone()

                aug_shape = aug_image.shape[-2:]
                # normalize the box coordinates
                aug_bbox[0::2] /= aug_shape[1]
                aug_bbox[1::2] /= aug_shape[0]
                boxes.append(aug_bbox)
                aug_shapes.append(aug_shape)

                template = self.get_template(
                    aug_image, aug_bbox_raw
                )  # crop the template from the first frame
                template_box = aug_bbox_raw

            elif self.is_train:
                bbox_ann = {
                    "bbox": [bboxes[frame_idx]],
                    "bbox_mode": utils.BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                aug_bbox = utils.transform_instance_annotations(
                    bbox_ann, transforms, image_shape
                )["bbox"]
                aug_bbox = torch.FloatTensor(aug_bbox)  # .unsqueeze(0)
                aug_bbox_raw = aug_bbox.clone()

                aug_shape = aug_image.shape[-2:]
                # normalize the box coordinates
                aug_bbox[0::2] /= aug_shape[1]
                aug_bbox[1::2] /= aug_shape[0]
                boxes.append(aug_bbox)
                aug_shapes.append(aug_shape)

                if frame_idx == ref_frame:
                    template = self.get_template(
                        aug_image, aug_bbox_raw
                    )  # crop the template from the first frame
                    template_box = aug_bbox_raw

            else:
                aug_shape = aug_image.shape[-2:]
                aug_shapes.append(aug_shape)

                if len(bboxes) > 1:
                    bbox_ann = {
                        "bbox": [bboxes[frame_idx]],
                        "bbox_mode": utils.BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                    aug_bbox = utils.transform_instance_annotations(
                        bbox_ann, transforms, image_shape
                    )["bbox"]
                    aug_bbox = torch.FloatTensor(aug_bbox)  # .unsqueeze(0)
                    aug_bbox_raw = aug_bbox.clone()

                    # aug_shape = aug_image.shape[-2:]
                    # normalize the box coordinates
                    aug_bbox[0::2] /= aug_shape[1]
                    aug_bbox[1::2] /= aug_shape[0]
                    boxes.append(aug_bbox)

                else:
                    boxes.append(torch.FloatTensor([0, 0, 0, 0]))

        images = ImageList.from_tensors(images)  # torch.stack(images, dim=0)
        bboxes = torch.stack(boxes, dim=0)
        aug_shapes = torch.FloatTensor(aug_shapes)  # T, 2

        """os.makedirs("./vis", exist_ok=True)
        # visualize the augmented boxes on augmented images
        visualize_boxes(
            images.tensor[0],
            bboxes[0:1],
            os.path.join(
                "./vis",
                os.path.basename(os.path.dirname(selected_file_names[0]))
                + "_"
                + os.path.basename(selected_file_names[0]),
            ),
            aug_shapes[0][0],
            aug_shapes[0][1],
        )

        visualize_boxes(
            images.tensor[1],
            bboxes[1:2],
            os.path.join(
                "./vis",
                os.path.basename(os.path.dirname(selected_file_names[0]))
                + "_"
                + os.path.basename(selected_file_names[1]),
            ),
            aug_shapes[1][0],
            aug_shapes[1][1],
        )"""

        text, _ = self.tokenize(
            ["something"] * (bboxes.shape[0] - 1),
            bboxes[1:].tolist(),
            height=None,
            width=None,
            tokenize_prompt=False,
        )
        text["input_ids"].squeeze_()
        text["attention_mask"].squeeze_()

        # set the prompt
        if (
            self.is_train
            and self.language_in_prompt
            and ann["template_query_score"] > self.score_threshold
            and random.random() > self.p_use_language
        ):
            prompt_text = task + ". " + ann["blip2query"]

        elif (
            not self.is_train
            and self.language_in_prompt
            and ann["template_query_score"] > self.score_threshold
        ):
            prompt_text = task + ". " + ann["blip2query"]

        else:
            prompt_text = task + ". "

        # set the prompt box
        if self.box_in_prompt:
            prompt_box = bboxes[:1].tolist()

        else:
            prompt_box = None

        prompt, _ = self.tokenize(
            [prompt_text],
            prompt_box,
            height=None,
            width=None,
            tokenize_prompt=True,
        )

        """if self.box_in_prompt and self.language_in_prompt:
            if self.is_train and ann["template_query_score"] > self.score_threshold and random.random() > self.p_use_language:
                prompt, _ = self.tokenize(
                    [task + ". " + ann["blip2query"]],
                    bboxes[:1].tolist(),
                    None,
                    None,
                    tokenize_prompt=True,
                )
            elif 
        
        elif self.box_in_prompt:
            prompt, _ = self.tokenize(
                [task + ". "],
                bboxes[:1].tolist(),
                None,
                None,
                tokenize_prompt=True,
            )
        
        elif self.language_in_prompt:
            prompt, _ = self.tokenize(
                [task + ". " + ann["blip2query"]],
                None,
                None,
                None,
                tokenize_prompt=True,
            )

        else:
            prompt, _ = self.tokenize(
                [task + ". "],
                None,
                None,
                None,
                tokenize_prompt=True,
            )"""

        prompt["input_ids"].squeeze_()
        prompt["attention_mask"].squeeze_()

        sample = {
            "task": task,
            "video": images.tensor,
            "prompt_box": template_box,
            "text": text,
            "text2": -1,
            "gt_boxes": bboxes[1:],  # T-1, 4
            "prompt": prompt,
            "image_sizes": aug_shapes,
            "height": height,
            "width": width,
            "name": ann["video_name"],
            "blip2query": ann["blip2query"] if "blip2query" in ann else "",
            "template_query_score": ann["template_query_score"]
            if "template_query_score" in ann
            else 0.0,
            "file_names": selected_file_names,
            "length": video_length,
            "target_token_weights": -1,
            "class_id": -1,
        }

        return sample
