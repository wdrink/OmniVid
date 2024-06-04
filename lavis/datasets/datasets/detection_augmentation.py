import numpy as np
import logging
import sys
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data import transforms as T
import copy


class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR,
        clip_frame_cnt=1,
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in [
            "range",
            "choice",
            "range_by_clip",
            "choice_by_clip",
        ], sample_style

        self.is_range = "range" in sample_style
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(
                    self.short_edge_length[0], self.short_edge_length[1] + 1
                )
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0  # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(
        self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1
    ):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0  # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


def build_augmentation(
    is_train,
    sampling_frame_num,
    crop_enabled,
    crop_type,
    crop_size,
    min_size_train,
    max_size_train,
    min_size_test,
    max_size_test,
    random_flip,
    augmentations,
):
    # crop_type = "absolute_range"
    # crop_size = (384, 600)
    # min_size_train = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # min_size_train = [490, 518, 546, 574, 602, 630, 658, 686, 714, 742, 770]
    # max_size_train = 1333

    # min_size_test = 480
    # max_size_test = 1333

    min_size_train_sampling = "choice_by_clip"
    # random_flip = "flip_by_clip"

    aug_list = []
    if is_train:
        # Crop
        if crop_enabled:
            aug_list.append(T.RandomCrop(crop_type, crop_size))

        # Resize
        min_size = min_size_train
        max_size = max_size_train
        sample_style = min_size_train_sampling
        ms_clip_frame_cnt = (
            sampling_frame_num if "by_clip" in min_size_train_sampling else 1
        )
        aug_list.append(
            ResizeShortestEdge(
                min_size,
                max_size,
                sample_style,
                clip_frame_cnt=ms_clip_frame_cnt,
            )
        )

        # Flip
        if random_flip != "none":
            if random_flip == "flip_by_clip":
                flip_clip_frame_cnt = sampling_frame_num
            else:
                flip_clip_frame_cnt = 1

            aug_list.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlip(
                    horizontal=(random_flip == "horizontal")
                    or (random_flip == "flip_by_clip"),
                    vertical=random_flip == "vertical",
                    clip_frame_cnt=flip_clip_frame_cnt,
                )
            )

        # Additional augmentations : brightness, contrast, saturation, rotation
        # augmentations = cfg.INPUT.AUGMENTATIONS
        if "brightness" in augmentations:
            aug_list.append(T.RandomBrightness(0.9, 1.1))
        if "contrast" in augmentations:
            aug_list.append(T.RandomContrast(0.9, 1.1))
        if "saturation" in augmentations:
            aug_list.append(T.RandomSaturation(0.9, 1.1))
        if "rotation" in augmentations:
            aug_list.append(
                T.RandomRotation(
                    [-15, 15],
                    expand=False,
                    center=[(0.4, 0.4), (0.6, 0.6)],
                    sample_style="range",
                )
            )
        if not crop_enabled:
            return aug_list
        else:
            aug_no_crop = copy.deepcopy(aug_list)
            del aug_no_crop[0]
            return aug_no_crop, aug_list
    else:
        # Resize
        min_size = min_size_test
        max_size = max_size_test
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    return aug_list
