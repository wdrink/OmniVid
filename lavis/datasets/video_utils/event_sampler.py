import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from lavis.tasks.eval_utils import visualize_video


def rand_apply(func_a, func_b, p):
    """Randomly apply function func_a or func_b with probability p."""
    return func_a if torch.rand(1) < p else func_b


def random_padding(
    sequence,
    saliency_label,
    duration,
    captions,
    timestamps,
    fps,
    preserve,
    pad_left,
    safe_range=0.0,
):
    start = [ts[0] for ts in timestamps]
    end = [ts[1] for ts in timestamps]

    if pad_left:
        # crop the right side and pad the left side
        if preserve:
            if safe_range > 0:
                max_offset = max(end) + safe_range
            else:
                max_offset = max(start) + 1
        else:
            max_offset = min(start) + 1

        max_offset = min(max_offset, duration)
        offset = random.uniform(max_offset, duration)
        offset = duration - offset
        offset_in_frame = int(offset * fps)

        # crop offset seconds from right and pad to left
        sequence = np.concatenate(
            (sequence[-offset_in_frame:], sequence[:-offset_in_frame]),
            axis=0,
        )
        if saliency_label is None:
            saliency_label = None
        else:
            saliency_label = np.concatenate(
                (
                    saliency_label[-offset_in_frame:],
                    saliency_label[:-offset_in_frame],
                ),
                axis=0,
            )

        # update the start/end/caption
        start = [s + offset for s in start]
        end = [min(e + offset, duration) for e in end]
        captions = captions

    else:
        # crop the left side and pad the right side
        if preserve:
            if safe_range > 0:
                max_offset = min(start) - safe_range
            else:
                max_offset = min(end) - 1
        else:
            max_offset = max(end) - 1

        max_offset = max(max_offset, 0)
        offset = random.uniform(0, max_offset)
        offset_in_frame = int(offset * fps)

        # crop offset seconds from left and pad to right
        sequence = np.concatenate(
            (sequence[offset_in_frame:], sequence[:offset_in_frame]),
            axis=0,
        )
        if saliency_label is None:
            saliency_label = None
        else:
            saliency_label = np.concatenate(
                (
                    saliency_label[offset_in_frame:],
                    saliency_label[:offset_in_frame],
                ),
                axis=0,
            )

        # update the start/end/caption
        start = [max(s - offset, 0) for s in start]
        end = [e - offset for e in end]
        captions = captions

    return sequence, saliency_label, duration, captions, start, end


def random_crop(
    sequence,
    saliency_label,
    duration,
    captions,
    timestamps,
    fps,
    preserve,
    threshold=1e4,
    safe_range=0.0,
    avoid_first_last=False,
    num_bins=300,
):
    """Randomly crops a sequence and its corresponding captions.
    args:
        sequence: A tensor of shape [T, H, W, C].
        duration: float, the duration of the sequence in seconds.
        captions: A list of N strings containing the captions.
        timestamps: A list of N tuples containing the start and end times of the captions.
        fps: float, the frame rate of the sequence.
        preserve: bool, whether to preserve all the events.
    """
    start = [ts[0] for ts in timestamps]
    end = [ts[1] for ts in timestamps]

    min_start = min(start)
    max_end = max(end)

    if min_start < threshold:
        (
            sequence,
            saliency_label,
            duration,
            captions,
            start,
            end,
        ) = random_padding(
            sequence,
            saliency_label,
            duration,
            captions,
            timestamps,
            fps,
            preserve,
            pad_left=True,
            safe_range=safe_range,
        )
    elif duration - max_end < threshold:
        (
            sequence,
            saliency_label,
            duration,
            captions,
            start,
            end,
        ) = random_padding(
            sequence,
            saliency_label,
            duration,
            captions,
            timestamps,
            fps,
            preserve,
            pad_left=False,
            safe_range=safe_range,
        )

    else:
        # sample a random offset_start
        if preserve:
            if safe_range > 0:
                max_offset = min(start) - safe_range
            else:
                max_offset = min(end) - 1
        else:
            max_offset = max(end) - 1

        max_offset = max(max_offset, 0)
        offset_start = random.uniform(
            0, max_offset
        )  # randomly sample a start point

        # Modify captions/start/end given the sampled offset_start:
        start = [s - offset_start for s in start]
        end = [e - offset_start for e in end]
        idx_to_keep = [i for i, e in enumerate(end) if e > 0]
        captions = [captions[i] for i in idx_to_keep]
        start = [max(start[i], 0) for i in idx_to_keep]
        end = [end[i] for i in idx_to_keep]

        offset_start_in_frame = int(offset_start * fps)
        sequence = sequence[offset_start_in_frame:]
        if saliency_label is not None:
            saliency_label = saliency_label[offset_start_in_frame:]

        # sample a random offset_end
        if preserve:
            if safe_range > 0:
                min_offset = max(end) + safe_range
            else:
                min_offset = max(start) + 1
        else:
            min_offset = min(start) + 1

        min_offset = min(min_offset, duration - offset_start)
        offset_end = random.uniform(
            min_offset, duration - offset_start
        )  # randomly sample an end point

        # Modify captions/start/end given the sampled offset_end:
        idx_to_keep = [i for i, s in enumerate(start) if s < offset_end]
        captions = [captions[i] for i in idx_to_keep]
        start = [start[i] for i in idx_to_keep]
        end = [min(end[i], offset_end) for i in idx_to_keep]

        offset_end_in_frame = int(offset_end * fps)
        sequence = sequence[: offset_end_in_frame + 1]

        if saliency_label is not None:
            saliency_label = saliency_label[: offset_end_in_frame + 1]

        duration = offset_end

    # finally we need to check whether the start and end are within the safe range
    if min(start) < safe_range and avoid_first_last:
        # we pad some zeros to the left
        pad_len = random.uniform(0, safe_range - min(start))
        pad_len_in_frame = math.ceil(pad_len * fps)
        sequence = np.concatenate(
            (
                np.zeros(
                    (pad_len_in_frame, *sequence.shape[1:]), dtype=np.uint8
                ),
                sequence,
            ),
            axis=0,
        )
        if saliency_label is None:
            saliency_label = None
        else:
            saliency_label = np.concatenate(
                (
                    np.zeros((pad_len_in_frame, *saliency_label.shape[1:])),
                    saliency_label,
                ),
                axis=0,
            )
        duration = duration + pad_len

        start = [s + pad_len for s in start]
        end = [e + pad_len for e in end]

    return sequence, saliency_label, duration, captions, start, end


def random_crop2(
    sequence,
    saliency_label,
    duration,
    captions,
    timestamps,
    fps,
    preserve,
    threshold=1e4,
    safe_range=0.0,
    avoid_first_last=False,
    num_bins=300,
):
    """Randomly crops a sequence and its corresponding captions.
    args:
        sequence: A tensor of shape [T, H, W, C].
        duration: float, the duration of the sequence in seconds.
        captions: A list of N strings containing the captions.
        timestamps: A list of N tuples containing the start and end times of the captions.
        fps: float, the frame rate of the sequence.
        preserve: bool, whether to preserve all the events.
    """
    start = [ts[0] for ts in timestamps]
    end = [ts[1] for ts in timestamps]

    # we have to make sure the start and end bins both follow a uniform distribution
    # to achieve, we first randomly sample a start bin
    sampled_start_bin = random.randint(0, num_bins - 1)

    # next, we perform random cropping / padding based on the sampled bins
    cur_start_bin = min(start) / duration * (num_bins - 1)

    if cur_start_bin < sampled_start_bin:
        # we pad the left side (influence both the start and end bin)
        offset = (sampled_start_bin - cur_start_bin) * duration / (num_bins - 1)
        offset_in_frame = math.ceil(offset * fps)

        # randomly sample offset_in_frame frames from the beginning and pad to the left
        min_frame_index = int(min(start) * fps)
        sequence = np.concatenate(
            (
                sequence[:min_frame_index],
                np.repeat(
                    sequence[min_frame_index : min_frame_index + 1],
                    offset_in_frame,
                    axis=0,
                ),
                sequence[min_frame_index:],
            ),
            axis=0,
        )
        if saliency_label is None:
            saliency_label = None
        else:
            saliency_label = np.concatenate(
                (
                    saliency_label[:min_frame_index],
                    np.repeat(
                        saliency_label[min_frame_index : min_frame_index + 1],
                        offset_in_frame,
                        axis=0,
                    ),
                    saliency_label[min_frame_index:],
                ),
                axis=0,
            )
        duration = duration + offset

        start = [s + offset for s in start]
        end = [e + offset for e in end]

    else:
        # we crop the left side (influence both the start and end bin)
        offset = (cur_start_bin - sampled_start_bin) * duration / (num_bins - 1)
        offset_in_frame = math.floor(offset * fps)

        sequence = sequence[offset_in_frame:]
        if saliency_label is None:
            saliency_label = None
        else:
            saliency_label = saliency_label[offset_in_frame:]

        duration = duration - offset

        start = [s - offset for s in start]
        end = [e - offset for e in end]

        # filter out the invalid segments
        idx_to_keep = [i for i, s in enumerate(start) if s >= 0]
        captions = [captions[i] for i in idx_to_keep]
        start = [start[i] for i in idx_to_keep]
        end = [end[i] for i in idx_to_keep]

    return sequence, saliency_label, duration, captions, start, end


def no_crop(
    sequence,
    saliency_label,
    duration,
    captions,
    timestamps,
    fps,
    preserve,
    threshold=1e4,
    safe_range=0.0,
    avoid_first_last=False,
    num_bins=300,
):
    """Returns the sequence and its corresponding captions without cropping."""
    return (
        sequence,
        saliency_label,
        duration,
        captions,
        [ts[0] for ts in timestamps],
        [ts[1] for ts in timestamps],
    )


def sample_equal_sequence(
    batch,
    num_steps,
    is_training,
    p,
    fps,
    preserve=True,
    threshold=1e4,
    safe_range=0.0,
    avoid_first_last=False,
    use_crop1=True,
    num_bins=300,
):
    sequence = batch["sequence"]
    duration = batch["duration"]
    captions = batch["captions"]
    timestamps = batch["timestamps"]
    saliency_label = batch.get("saliency_label", None)
    clip_captions = batch.get("clip_captions", None)
    clip_timestamps = batch.get("clip_timestamps", None)
    raw_duration = duration

    # clamp the timestamps
    start = [max(ts[0], 0) for ts in timestamps]
    end = [min(ts[1], duration) for ts in timestamps]
    timestamps = [(s, e) for s, e in zip(start, end)]

    """Samples at equal distance num_steps features + pad + random temporal crop."""
    if is_training and p > 0:
        if use_crop1:
            (
                sequence,
                saliency_label,
                duration,
                captions,
                start,
                end,
            ) = rand_apply(random_crop, no_crop, p,)(
                sequence,
                saliency_label,
                duration,
                captions,
                timestamps,
                fps=fps,
                preserve=preserve,
                threshold=threshold,
                safe_range=safe_range,
                avoid_first_last=avoid_first_last,
                num_bins=num_bins,
            )
        else:
            (
                sequence,
                saliency_label,
                duration,
                captions,
                start,
                end,
            ) = rand_apply(random_crop2, no_crop, p,)(
                sequence,
                saliency_label,
                duration,
                captions,
                timestamps,
                fps=fps,
                preserve=preserve,
                threshold=threshold,
                safe_range=safe_range,
                avoid_first_last=avoid_first_last,
                num_bins=num_bins,
            )

    else:
        sequence, saliency_label, duration, captions, start, end = no_crop(
            sequence,
            saliency_label,
            duration,
            captions,
            timestamps,
            fps=fps,
            preserve=preserve,
            threshold=threshold,
            safe_range=safe_range,
            avoid_first_last=avoid_first_last,
            num_bins=num_bins,
        )

    # pad or sample
    sequence_length, H, W, _ = sequence.shape
    if sequence_length < num_steps:
        # pad
        zeros = np.zeros(
            (num_steps - sequence_length, H, W, 3),
            dtype=np.uint8,
        )
        raw_sample_frms = np.concatenate((sequence, zeros), axis=0)
        if saliency_label is None:
            raw_sample_saliency_label = None
        else:
            raw_sample_saliency_label = np.concatenate(
                (saliency_label, np.zeros((num_steps - sequence_length))),
                axis=0,
            )
        # generate masks for the padded frames
        raw_sample_masks = np.zeros((num_steps,), dtype=np.uint8)
        raw_sample_masks[:sequence_length] = 1
        # print("padding length is ", num_steps - sequence_length)

    else:
        # linearly sample
        frame_indices = np.linspace(0, sequence_length - 1, num_steps).astype(
            int
        )
        raw_sample_frms = sequence[frame_indices]
        if saliency_label is None:
            raw_sample_saliency_label = None
        else:
            raw_sample_saliency_label = saliency_label[frame_indices]
        raw_sample_masks = np.ones((num_steps,), dtype=np.uint8)

    # Correct dimensions and types.
    duration = round(duration, 2)
    duration = min(duration, raw_duration)

    start = [round(s, 2) for s in start]
    end = [round(e, 2) for e in end]

    # convert start and end time back to timestamps
    timestamps = [(s, e) for s, e in zip(start, end)]

    updated_batch = {
        "sequence": raw_sample_frms,
        "sequence_mask": raw_sample_masks,
        "saliency_label": raw_sample_saliency_label,
        "duration": duration,
        "captions": captions,
        "timestamps": timestamps,
        "clip_captions": clip_captions,
        "clip_timestamps": clip_timestamps,
    }
    return updated_batch
