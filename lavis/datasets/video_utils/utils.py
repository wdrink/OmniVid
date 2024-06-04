import io
import json
import math
import numbers
import os
import random
import re
from enum import Enum

import numpy as np
from PIL import Image

# import lavis.common.dist_utils as dist_utils
import mmcv
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
from decord import VideoReader
from decord import cpu, gpu
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torch.nn.functional import interpolate as img_tensor_resize
from torch.nn.functional import pad as img_tensor_pad
from torch.nn.modules.utils import _quadruple
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


class VideoNorm(object):
    """Apply Normalization to Image Pixels on GPU"""

    def __init__(
        self,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ):
        # self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (N, 3, H, W)
        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.0)
        re = img.sub_(self.mean).div_(self.std)
        return re




class VideoResizeSquare(object):
    def __init__(self, out_size, interpolation="nearest"):
        assert isinstance(out_size, int)
        self.out_size = out_size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be scaled.

        Returns:
            torch.tensor: Rescaled video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                t, h, w, c = video.shape
                assert (
                    c == 3
                ), "Expecting 3-channel color video, got video of shape {}".format(
                    video.shape
                )
            else:
                raise RuntimeError(
                    "Expecting 4-dimensional tensor of shape (b,t,h,w), got {}".format(
                        video.shape
                    )
                )

            # t, h, w, c -> t, c, h, w
            video = video.permute(0, 3, 1, 2)
            short_side = h if h < w else w
            resized_video = img_tensor_resize(
                video,
                size=((self.out_size, self.out_size)),
                mode=self.interpolation,
            )

            # t, c, h, w -> t, h, w, c
            return resized_video.permute(0, 2, 3, 1)

        else:
            # in other data class, the order of shape might be different.
            raise NotImplementedError(
                "Support only torch.Tensor as input, got {}".format(type(video))
            )

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.out_size, self.interpolation
        )



def load_video_from_bytes_resample(
    video_stream, height=None, width=None, fps=-1, return_fps=False
):
    if not height or not width:
        vr = VideoReader(video_stream)
    else:
        vr = VideoReader(video_stream, width=width, height=height, num_threads=1)

    default_fps = vr.get_avg_fps()
    if default_fps <= fps:
        fps = -1

    if fps != -1:
        # resample the video to the specified fps
        duration = len(vr) / default_fps
        num_frames_to_sample = int(duration * fps)
        resample_indices = np.linspace(
            0, len(vr) - 1, num_frames_to_sample
        ).astype(int)

        sampled_frms = np.array(vr.get_batch(resample_indices)).astype(
            np.uint8
        )
        default_fps = fps

    else:
        sampled_frms = np.array(
            vr.get_batch(np.arange(0, len(vr), 1, dtype=int))
        ).astype(np.uint8)

    if return_fps:
        return sampled_frms, default_fps
    else:
        return sampled_frms

    

def load_video_from_frames(
    frames_path, height, width, fps, default_fps, default_duration
):
    frames = []
    for frame in sorted(os.listdir(frames_path)):
        frame_path = os.path.join(frames_path, frame)
        frame = mmcv.imread(frame_path)
        if height and width:
            frame = mmcv.imresize(frame, (width, height))
        frames.append(frame)

    frames = np.stack(frames)

    if fps != -1:
        num_frames_saved = len(frames)  # duration * default_fps
        duration = int(num_frames_saved / default_fps) + 1
        resample_indices = np.linspace(
            0, num_frames_saved - 1, duration
        ).astype(int)
        sampled_frms = frames[resample_indices]

    else:
        sampled_frms = frames
    
    return sampled_frms


def load_video_from_path_tvio(
    video_path, 
    frm_sampling_strategy,
    height=None,
    width=None,
    num_frm=None,
    return_fps=False,
    spatial_crop=None,
    temporal_crop=None
):
    video = tv.io.read_video(video_path, pts_unit="sec")
    if not height or not width:
        sampled_frms = np.array(video[0])
    elif spatial_crop is None:
        # T, H, W, C
        sampled_frms_tensor = video[0]
        # expected: t, c, h, w 
        resize_func = VideoResizeSquare(out_size=height)
        sampled_frms_tensor = resize_func(sampled_frms_tensor)
        sampled_frms = np.array(sampled_frms_tensor)
    else:
        sampled_frms_tensor = video[0]
        resize_func = VideoResizeSquare(out_size=int(height*1.5))
        sampled_frms_tensor = resize_func(sampled_frms_tensor)
        sampled_frms = np.array(sampled_frms_tensor)
    
    specified_num_frm = num_frm    
    default_fps = video[2]["video_fps"]
    vlen = sampled_frms.shape[0]

    if num_frm is None:
        num_frm = vlen

    num_frm = min(num_frm, vlen)

    if frm_sampling_strategy == "uniform":
        if temporal_crop is None:
            frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)
        else:
            ind, total = temporal_crop.split("/")
            ind = int(ind)
            total = int(total)
            all_indices = np.linspace(0, vlen - 1, total * num_frm).astype(int)
            segs = len(all_indices) // total

            if segs == num_frm:
                if ind == total - 1:
                    frame_indices = all_indices[-segs: ]
                else:
                    frame_indices = all_indices[(ind - 1) * segs: ind * segs]
            else:
                frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)

    elif frm_sampling_strategy == "rand":
        frame_indices = sorted(random.sample(range(vlen), num_frm))

    elif frm_sampling_strategy == "all":
        frame_indices = np.arange(0, vlen).astype(int)

    else:
        raise NotImplementedError(
            "Invalid sampling strategy {} ".format(frm_sampling_strategy)
        )

    raw_sample_frms = sampled_frms[
        frame_indices
    ]

    if spatial_crop is None:
        raw_sample_frms = raw_sample_frms
    elif spatial_crop == "top_left":
        raw_sample_frms = raw_sample_frms[:, :height, :width]
    elif spatial_crop == "center":
        raw_sample_frms = raw_sample_frms[:, int(height * 0.25): int(height * 0.25) + height, int(width * 0.25): int(width * 0.25) + width]
    elif spatial_crop == "bottom_right":
        raw_sample_frms = raw_sample_frms[:, -height:, -width:]
    else:
        raise NotImplementedError

    if specified_num_frm is None:
        masks = np.ones(len(raw_sample_frms), dtype=np.uint8)

    # pad the video if the number of frames is less than specified
    elif len(raw_sample_frms) < specified_num_frm:
        prev_length = len(raw_sample_frms)
        zeros = np.zeros(
            (specified_num_frm - prev_length, height, width, 3),
            dtype=np.uint8,
        )
        raw_sample_frms = np.concatenate((raw_sample_frms, zeros), axis=0)
        masks = np.zeros(specified_num_frm, dtype=np.uint8)
        masks[:prev_length] = 1

    else:
        masks = np.ones(specified_num_frm, dtype=np.uint8)

    if return_fps:
        return raw_sample_frms, masks, default_fps
    else:
        return raw_sample_frms, masks


def load_video_from_path_decord(
    video_path,
    frm_sampling_strategy,
    height=None,
    width=None,
    start_time=None,
    end_time=None,
    fps=-1,
    num_frm=None,
    return_fps=False,
    spatial_crop=None,
    temporal_crop=None
):
    specified_num_frm = num_frm
    if not height or not width:
        vr = VideoReader(video_path)
    elif spatial_crop is None:
        vr = VideoReader(video_path, width=width, height=height)
    else:
        vr = VideoReader(video_path, width=int(width*1.5), height=int(height*1.5))
    
    default_fps = vr.get_avg_fps()
    if default_fps <= fps:
        fps = -1

    if fps != -1:
        # resample the video to the specified fps
        duration = len(vr) / default_fps
        num_frames_to_sample = int(duration * fps)
        resample_indices = np.linspace(
            0, len(vr) - 1, num_frames_to_sample
        ).astype(int)
        sampled_frms = np.array(vr.get_batch(resample_indices)).astype(
            np.uint8
        )
        default_fps = fps

    else:
        sampled_frms = np.array(
            vr.get_batch(np.arange(0, len(vr), 1, dtype=int))
        ).astype(np.uint8)

    vlen = sampled_frms.shape[0]

    if num_frm is None:
        num_frm = vlen

    num_frm = min(num_frm, vlen)

    if start_time or end_time:
        assert (
            fps > 0
        ), "must provide video fps if specifying start and end time."
        start_idx = min(int(start_time * fps), vlen)
        end_idx = min(int(end_time * fps), vlen)

    else:
        start_idx, end_idx = 0, vlen

    if frm_sampling_strategy == "uniform":
        if temporal_crop is None:
            frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)
        else:
            ind, total = temporal_crop.split("/")
            ind = int(ind)
            total = int(total)
            all_indices = np.linspace(0, vlen - 1, total * num_frm).astype(int)
            segs = len(all_indices) // total

            if segs == num_frm:
                if ind == total - 1:
                    frame_indices = all_indices[-segs: ]
                else:
                    frame_indices = all_indices[(ind - 1) * segs: ind * segs]
            else:
                frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)

    elif frm_sampling_strategy == "nlvl_uniform":
        frame_indices = np.arange(
            start_idx, end_idx, vlen / num_frm
        ).astype(int)

    elif frm_sampling_strategy == "nlvl_rand":
        frame_indices = np.arange(
            start_idx, end_idx, vlen / num_frm
        ).astype(int)

        strides = [
            frame_indices[i] - frame_indices[i - 1]
            for i in range(1, len(frame_indices))
        ] + [vlen - frame_indices[-1]]
        pertube = np.array(
            [np.random.randint(0, stride) for stride in strides]
        )

        frame_indices = frame_indices + pertube

    elif frm_sampling_strategy == "rand":
        frame_indices = sorted(random.sample(range(vlen), num_frm))

    elif frm_sampling_strategy == "headtail":
        frame_indices_head = sorted(
            random.sample(range(vlen // 2), num_frm // 2)
        )
        frame_indices_tail = sorted(
            random.sample(range(vlen // 2, vlen), num_frm // 2)
        )
        frame_indices = frame_indices_head + frame_indices_tail

    elif frm_sampling_strategy == "all":
        frame_indices = np.arange(0, vlen).astype(int)

    else:
        raise NotImplementedError(
            "Invalid sampling strategy {} ".format(frm_sampling_strategy)
        )

    raw_sample_frms = sampled_frms[
        frame_indices
    ]

    if spatial_crop is None:
        raw_sample_frms = raw_sample_frms
    elif spatial_crop == "top_left":
        raw_sample_frms = raw_sample_frms[:, :height, :width]
    elif spatial_crop == "center":
        raw_sample_frms = raw_sample_frms[:, int(height * 0.25): int(height * 0.25) + height, int(width * 0.25): int(width * 0.25) + width]
    elif spatial_crop == "bottom_right":
        raw_sample_frms = raw_sample_frms[:, -height:, -width:]
    else:
        raise NotImplementedError

    if specified_num_frm is None:
        masks = np.ones(len(raw_sample_frms), dtype=np.uint8)

    # pad the video if the number of frames is less than specified
    elif len(raw_sample_frms) < specified_num_frm:
        prev_length = len(raw_sample_frms)
        zeros = np.zeros(
            (specified_num_frm - prev_length, height, width, 3),
            dtype=np.uint8,
        )
        raw_sample_frms = np.concatenate((raw_sample_frms, zeros), axis=0)
        masks = np.zeros(specified_num_frm, dtype=np.uint8)
        masks[:prev_length] = 1

    else:
        masks = np.ones(specified_num_frm, dtype=np.uint8)

    if return_fps:
        return raw_sample_frms, masks, default_fps
    else:
        return raw_sample_frms, masks


def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.
    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim (bool): If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`. Default: ``True``
    Returns:
        torch.Tensor: tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError(
            "Input type must be a numpy.ndarray. Got {}".format(type(image))
        )

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array"
        )

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape)
        )

    return tensor.unsqueeze(0) if not keepdim else tensor


def get_padding(image, max_w, max_h, pad_all=False):
    # keep the images to upper-left corner
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_padding, v_padding = max_w - w, max_h - h
    if pad_all:
        h_padding /= 2
        v_padding /= 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    else:
        l_pad, t_pad = 0, 0
        r_pad, b_pad = h_padding, v_padding
    if isinstance(image, torch.Tensor):
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    else:
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class ImagePad(object):
    def __init__(self, max_w, max_h, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.max_w = max_w
        self.max_h = max_h
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        if isinstance(img, torch.Tensor):
            paddings = _quadruple(get_padding(img, self.max_w, self.max_h))
            return img_tensor_pad(img, paddings, self.padding_mode, self.fill)
        return img_pad(
            img,
            get_padding(img, self.max_w, self.max_h),
            self.fill,
            self.padding_mode,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )


def get_resize_size(image, max_size):
    """
    Args:
        image: PIL Image or torch.tensor
        max_size:

    Returns:

    Note the height/width order difference
    >>> pil_img = Image.open("raw_img_tensor.jpg")
    >>> pil_img.size
    (640, 480)  # (width, height)
    >>> np_img = np.array(pil_img)
    >>> np_img.shape
    (480, 640, 3)  # (height, width, 3)
    """
    # note the order of height and width for different inputs
    if isinstance(image, torch.Tensor):
        # width, height = image.shape[-2:]
        height, width = image.shape[-2:]
    else:
        width, height = image.size

    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))
    return size


class VideoRandomSquareCrop(object):
    def __init__(self, crop_size, p=0.5):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        self.p = p

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be cropped.

        Returns:
            torch.tensor: cropped video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                b, t, h, w = video.shape
            else:
                raise RuntimeError(
                    "Expecting 4-dimensional tensor of shape (b,t,h,w), got {}".format(
                        video.shape
                    )
                )

            # if random.uniform(0, 1) < self.p:
            #     video = torch.flip(video, (3,))

            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, :, x : x + self.crop_size, y : y + self.crop_size]

        else:
            raise NotImplementedError(
                "Support only torch.Tensor as input, got {}".format(type(video))
            )

class ImageResize(object):
    """Resize the input image (torch.tensor) to the given size.

    Args:
        max_size (int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_size, interpolation=Image.BILINEAR):
        assert isinstance(max_size, int)
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (torch.tensor): Image to be scaled.

        Returns:
            torch.tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            assert isinstance(self.interpolation, str)
            return img_tensor_resize(
                img,
                size=get_resize_size(img, self.max_size),
                mode=self.interpolation,
                align_corners=False,
            )
        return img_resize(
            img, get_resize_size(img, self.max_size), self.interpolation
        )

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.size, interpolate_str
        )


def get_imagenet_transform(min_size=600, max_size=1000):
    """parameters from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This simply crop the center square from the image
    """
    if min_size != 600:
        import warnings

        warnings.warn(
            f"Warning: min_size is not used in image transform, "
            f"setting min_size will have no effect."
        )
    return transforms.Compose(
        [
            ImageResize(
                max_size, Image.BILINEAR
            ),  # longer side will be resized to 1000
            ImagePad(max_size, max_size),  # pad to 1000 * 1000
        ]
    )


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU"""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 1, 3, 1, 1)
        # assert max(std) <= 1 and min(std) >= 0\
        #     or max(mean) <= 1 and min(mean) >= 0,\
        #         "Please provide mean or std within range [0, 1]"

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (B, N, 3, H, W)

        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.0)
        return img.sub_(self.mean).div_(self.std)


def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    """
    Args:
        examples: iterable, examples grouped by image/video
        chunk_size: int, number of examples in each chunk.
        pad_to_divisible: bool, pad the examples to be divisible by chunk_size.
    >>> test_examples = [3, 4, 5, 6, 7]
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=True)
    [[3, 4], [5, 6], [7, 7]]  # the lst element has some randomness
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=False)
    [[3, 4], [5, 6], [7]]
    """
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i * chunk_size : (i + 1) * chunk_size])
    return chunked_examples


# def repeat_tensor_rows(raw_tensor, row_repeats):
#     """ repeat raw_tensor[i] row_repeats[i] times.
#     Args:
#         raw_tensor: (B, *)
#         row_repeats: list(int), len(row_repeats) == len(raw_tensor)
#     """
#     assert len(raw_tensor) == len(raw_tensor), "Has to be the same length"
#     if sum(row_repeats) == len(row_repeats):
#         return raw_tensor
#     else:
#         indices = torch.LongTensor(
#             flat_list_of_lists([[i] * r for i, r in enumerate(row_repeats)])
#         ).to(raw_tensor.device)
#         return raw_tensor.index_select(0, indices)


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        "",
        question.lower(),
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


"""def save_result(result, result_dir, filename, remove_duplicate=""):
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, dist_utils.get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    dist.barrier()

    if dist_utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(dist_utils.get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)

    return final_result_file"""


def coco_caption_eval(coco_gt_root, results_file, split):
    if "coco" in coco_gt_root:
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

    else:
        annotation_file = coco_gt_root

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


def mappixels(img, logit_laplace_eps=0.1):
    img = torch.true_divide(img, 255)
    img = (1 - 2 * logit_laplace_eps) * img + logit_laplace_eps
    return img


def copyresize(img, scale=None, interpolation="lanczos"):
    if scale is not None and not isinstance(scale, tuple):
        scale = (scale,) * 2

    if scale is not None:
        img = mmcv.imresize(img, scale, interpolation=interpolation)

    return img


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"
