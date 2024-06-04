import copy
import torch
import torch.nn.functional as F


def truncation_segment(segment, duration):
    return torch.clamp(segment, 0.0, duration)


def jitter_segment(
    segment, duration, min_range=0.0, max_range=0.05, truncation=True
):
    """Jitter the temporal segment.

    Args:
        segment: `torch.Tensor` of shape (n, 2), containing start and end times for each segment.
        duration: `float` scalar, the duration of the original sequence in seconds.
        min_range: min jitter range in ratio to segment duration.
        max_range: max jitter range in ratio to segment duration.
        truncation: whether to truncate resulting segments to remain within [0, duration].

    Note:
        To create noisy positives, set min_range=0, which enables a truncated normal
        distribution. max_range <=0.05: noisy duplicates, <=0.02: near duplicates.
        To create negatives: set min_range >= 0.1 to avoid false negatives;
        suggested max_range <=0.4 to avoid too much randomness.

    Returns:
        jittered temporal segments.
    """
    n = segment.size(0)
    t_start = segment[:, 0]
    t_end = segment[:, 1]

    # Calculate the duration of each segment
    segment_duration = t_end - t_start

    # Calculate the noise range based on min and max jitter ratios
    noise_range = torch.stack([segment_duration, segment_duration], dim=-1)
    if min_range == 0:
        noise_rate = torch.normal(
            mean=0.0, std=max_range / 2.0, size=(n, 2), dtype=segment.dtype
        )
    else:
        noise_rate1 = torch.rand((n, 2)) * (max_range - min_range) + min_range
        noise_rate2 = torch.rand((n, 2)) * (max_range - min_range) - max_range
        selector = (torch.rand((n, 2)) < 0.5).float()
        noise_rate = noise_rate1 * selector + noise_rate2 * (1.0 - selector)

    # Calculate the jittered segments
    jittered_segments = segment + noise_range * noise_rate

    return (
        truncation_segment(jittered_segments, duration)
        if truncation
        else jittered_segments
    )


def shift_segment(segment, duration, truncation=True):
    """Shifting temporal segments without changing the segment duration.

    Args:
        segment: `torch.Tensor` of shape (n, 2), containing start and end times for each segment.
        duration: `float` scalar, the duration of the original sequence in seconds.
        truncation: whether to truncate resulting segments to remain within [0, duration].

    Returns:
        shifted temporal segments.
    """
    n = segment.size(0)
    # randomly sample new segment centers
    center = torch.rand(n, 1)
    shift = (center - 0.5) * duration * 0.1  # 10% shift in either direction
    t_start = segment[:, :1] + shift
    t_end = segment[:, 1:] + shift

    # Combine the shifted start and end times into the new segments
    shifted_segments = torch.cat([t_start, t_end], dim=-1)

    return (
        truncation_segment(shifted_segments, duration)
        if truncation
        else shifted_segments
    )


def random_segment(n, duration, truncation=True):
    """Generating random n temporal segments with max size specified within [0, duration].

    Args:
        n: `int`, number of segments to generate.
        duration: `float` scalar, the duration of the original sequence in seconds.
        max_size: `float` scalar, maximum size of the generated segments in seconds.
        truncation: whether to truncate resulting segments to remain within [0, duration].

    Returns:
        Randomly generated temporal segments.
    """
    center_of_segments = torch.rand(n, 1) * duration
    duration_of_segments = torch.empty(n, 1).normal_(mean=0, std=duration / 2.0)

    # Ensure the generated segment is within the range [0, duration]
    t_start = torch.max(
        center_of_segments - torch.abs(duration_of_segments) / 2,
        torch.tensor(0.0),
    )
    t_end = torch.min(
        center_of_segments + torch.abs(duration_of_segments) / 2,
        torch.tensor(duration),
    )

    # Combine the start and end times into the new segments
    random_segments = torch.cat([t_start, t_end], dim=-1)

    return (
        truncation_segment(random_segments, duration)
        if truncation
        else random_segments
    )


def augment_segment(
    segment,
    segment_label,
    max_jitter,
    n_noise_segments,
    noise_label,
    rand_noise_label,
    duration,
    mix_rate=0.0,
):
    """Augment segments.

    There are two types of noises to add:
      1. Bad segments: jittered segments, shifted segments, or random segments.
      2. Duplicated segments.

    Args:
      segment: `float` tensor of shape (n, 2), ranged between 0 and `duration`.
      segment_label: `int` tensor of shape (n,).
      max_jitter: `float` scalar specifying max jitter range for positive segments.
      n_noise_segments: `int` scalar tensor specifying size of the extra noise to add.
      noise_label: `int` scalar, the label to assign to the added noise.
      mix_rate: `float`. Probability of injecting the bad segments in the middle of
        original segments, followed by dup segments at the end; otherwise simply append
        all noises at the end of original segments.
      duration: float, the duration of the video in seconds.

    Returns:
      segment_new: augmented segments that's `n_noise_segments` larger than the original.
      label_new: new label for segment_new.
      is_real: a `float` 0/1 indicator for whether a segment is real.
      is_noise: a `float` 0/1 indicator for whether a segment is extra.
    """
    n = segment.size(0)
    dup_segment_size = torch.randint(0, n_noise_segments + 1, []).item()
    dup_segment_size = 0 if n == 0 else dup_segment_size
    bad_segment_size = n_noise_segments - dup_segment_size
    multiplier = 1 if n == 0 else (n_noise_segments // n) + 1
    segment_tiled = segment.repeat(multiplier, 1)
    # print(f"{bad_segment_size} bad segments and {dup_segment_size} dup segments")

    # Create bad segments.
    segment_tiled = segment_tiled[torch.randperm(segment_tiled.size(0))]
    bad_segment_shift = shift_segment(
        segment_tiled[:bad_segment_size], duration=duration
    )
    bad_segment_random = random_segment(bad_segment_size, duration=duration)
    bad_segment = torch.cat([bad_segment_shift, bad_segment_random], 0)
    bad_segment = bad_segment[torch.randperm(bad_segment.size(0))][
        :bad_segment_size
    ]

    if not rand_noise_label:
        bad_segment_label = (
            torch.zeros([bad_segment_size], dtype=segment_label.dtype)
            + noise_label
        )
    else:
        bad_segment_label = torch.randint(
            0, noise_label, (bad_segment_size,), dtype=segment_label.dtype
        )

    # Create dup segments.
    segment_tiled = segment_tiled[torch.randperm(segment_tiled.size(0))]
    dup_segment = jitter_segment(
        segment_tiled[:dup_segment_size],
        min_range=0,
        max_range=0.1,
        duration=duration,
    )

    if not rand_noise_label:
        dup_segment_label = (
            torch.zeros([dup_segment_size], dtype=segment_label.dtype)
            + noise_label
        )
    else:
        dup_segment_label = torch.randint(
            0, noise_label, (dup_segment_size,), dtype=segment_label.dtype
        )

    # Jitter positive segments.
    if max_jitter > 0:
        segment = jitter_segment(
            segment, min_range=0, max_range=max_jitter, duration=duration
        )

    # create label masks to indicate which segments are real and which are noise: 1 are real and 0 are noise

    if torch.rand([]) < mix_rate:
        # Mix the segments with bad segments, appended by dup segments.
        segment_new = torch.cat([segment, bad_segment], 0)
        label_masks = torch.ones([n + bad_segment_size])
        label_masks[n:] = 0
        segment_new_label = torch.cat([segment_label, bad_segment_label], 0)
        idx = torch.randperm(segment_new.size(0))
        segment_new = segment_new[idx]
        label_masks = label_masks[idx]
        segment_new_label = segment_new_label[idx]
        segment_new = torch.cat([segment_new, dup_segment], 0)
        segment_new_label = torch.cat([segment_new_label, dup_segment_label], 0)
        label_masks = torch.cat([label_masks, torch.zeros([dup_segment_size])])

    else:
        # Merge bad segments and dup segments into noise segments.
        noise_segment = torch.cat([bad_segment, dup_segment], 0)
        noise_segment_label = torch.cat(
            [bad_segment_label, dup_segment_label], 0
        )

        if n_noise_segments > 0:
            idx = torch.randperm(n_noise_segments)
            noise_segment = noise_segment[idx]
            noise_segment_label = noise_segment_label[idx]

        # Append noise segments to segments and create mask.
        segment_new = torch.cat([segment, noise_segment], 0)
        segment_new_label = torch.cat([segment_label, noise_segment_label], 0)
        label_masks = torch.cat(
            [torch.ones([n]), torch.zeros([n_noise_segments])]
        )

    # print(mix_rate, label_masks)

    return segment_new, segment_new_label, label_masks


def inject_noise_segment(
    segments,
    categories,
    duration,
    max_instances_per_image,
    noise_label,
    rand_noise_label,
    mix_rate,
):
    segments = copy.deepcopy(segments)
    categories = copy.deepcopy(categories)
    label_masks = torch.ones([len(segments)])

    if not isinstance(segments, torch.Tensor):
        segments = torch.FloatTensor(segments)

    if not isinstance(categories, torch.Tensor):
        categories = torch.LongTensor(categories)

    assert len(segments) == len(categories)
    num_instances = len(segments)
    if num_instances < max_instances_per_image:
        n_noise_bbox = max_instances_per_image - num_instances
        segments, categories, label_masks = augment_segment(
            segments,
            categories,
            0.0,
            n_noise_bbox,
            noise_label,
            rand_noise_label,
            duration,
            mix_rate,
        )

    segments = segments.tolist()
    categories = categories.tolist()

    return segments, categories, label_masks


if __name__ == "__main__":
    duration = 80
    random_segments = random_segment(3, duration, truncation=True)
    random_labels = torch.randint(0, 10, (3,))

    print(random_segments, random_labels)

    noise_segments, noise_labels, label_masks = inject_noise_segment(
        random_segments,
        random_labels,
        duration,
        max_instances_per_image=10,
        noise_label=10,
        rand_noise_label=False,
        mix_rate=0.5,
    )
    print("noise_segments")
    print(noise_segments, noise_labels, label_masks)
