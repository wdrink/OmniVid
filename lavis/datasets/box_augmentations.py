import copy
import torch
import torch.nn.functional as F


def truncation_bbox(bbox):
    return torch.clamp(bbox, 0.0, 1.0)


def jitter_bbox(bbox, min_range=0.0, max_range=0.05, truncation=True):
    """Jitter the bbox.

    Args:
        bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
        min_range: min jitter range in ratio to bbox size.
        max_range: max jitter range in ratio to bbox size.
        truncation: whether to truncate resulting bbox to remain [0, 1].

    Note:
        To create noisy positives, set min_range=0, which enables truncated normal
        distribution. max_range <=0.05: noisy duplicates, <=0.02: near duplicate.
        To create negatives: set min_range >= 0.1 to avoid false negatives;
        suggested max_range <=0.4 to avoid too much randomness.

    Returns:
        jittered bbox.
    """
    if bbox.ndim == 3:
        # batched jitter_bbox
        B = bbox.size(0)
        bbox = bbox.view(-1, 4)
        is_batched = True
    else:
        is_batched = False

    n = bbox.size(0)
    h = bbox[:, 2] - bbox[:, 0]
    w = bbox[:, 3] - bbox[:, 1]
    noise = torch.stack([h, w, h, w], dim=-1).to(bbox.device)
    if min_range == 0:
        noise_rate = (torch.randn((n, 4)) * (max_range / 2.0)).to(bbox.device)
    else:
        noise_rate1 = torch.rand((n, 4)) * (max_range - min_range) + min_range
        noise_rate2 = torch.rand((n, 4)) * (max_range - min_range) - max_range
        selector = (torch.rand((n, 4)) < 0.5).float()
        noise_rate = (
            noise_rate1 * selector + noise_rate2 * (1.0 - selector)
        ).to(bbox.device)

    bbox = bbox + noise * noise_rate
    re = truncation_bbox(bbox) if truncation else bbox

    if is_batched:
        re = re.view(B, -1, 4)

    return re


def shift_bbox(bbox, truncation=True):
    """Shifting bbox without changing the bbox height and width."""
    n = bbox.size(0)
    # randomly sample new bbox centers.
    cy = torch.rand(n, 1)
    cx = torch.rand(n, 1)
    h = bbox[:, 2:3] - bbox[:, 0:1]
    w = bbox[:, 3:4] - bbox[:, 1:2]
    bbox = torch.cat(
        [
            cy - torch.abs(h) / 2,
            cx - torch.abs(w) / 2,
            cy + torch.abs(h) / 2,
            cx + torch.abs(w) / 2,
        ],
        dim=-1,
    )
    return truncation_bbox(bbox) if truncation else bbox


def random_bbox(n, max_size=1.0, truncation=True):
    """Generating random n bbox with max size specified within [0, 1]."""
    cy = torch.rand(n, 1)
    cx = torch.rand(n, 1)
    h = torch.empty(n, 1).normal_(mean=0, std=max_size / 2.0)
    w = torch.empty(n, 1).normal_(mean=0, std=max_size / 2.0)
    bbox = torch.cat(
        [
            cy - torch.abs(h) / 2,
            cx - torch.abs(w) / 2,
            cy + torch.abs(h) / 2,
            cx + torch.abs(w) / 2,
        ],
        dim=-1,
    )
    return truncation_bbox(bbox) if truncation else bbox


def augment_bbox(
    bbox, bbox_label, max_jitter, n_noise_bbox, noise_label, mix_rate=0.0
):
    """Augment bbox.

    There are two types of noises to add:
      1. Bad bbox: jittered bbox, shifted bbox, or random bbox.
      2. Duplicated bbox.

    Args:
      bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
      bbox_label: `int` tensor of shape (n,).
      max_jitter: `float` scalar specifying max jitter range for positive bbox.
      n_noise_bbox: `int` scalar tensor specifying size of the extra noise to add.
      mix_rate: `float`. Probability of injecting the bad bbox in the middle of
        original bbox, followed by dup bbox at the end; otherwise simply append
        all noises at the end of original bbox.

    Returns:
      bbox_new: augmented bbox that's `n_noise_bbox` larger than original.
      label_new: new label for bbox_new.
      is_real: a `float` 0/1 indicator for whether a bbox is real.
      is_noise: a `float` 0/1 indicator for whether a bbox is extra.
    """
    n = bbox.size(0)
    dup_bbox_size = torch.randint(0, n_noise_bbox + 1, [])
    dup_bbox_size = 0 if n == 0 else dup_bbox_size.item()
    bad_bbox_size = n_noise_bbox - dup_bbox_size
    multiplier = 1 if n == 0 else (n_noise_bbox // n) + 1
    bbox_tiled = bbox.repeat(multiplier, 1)
    print(f"{bad_bbox_size} bad bboxes and {dup_bbox_size} dup bboxes")

    # Create bad bbox.
    bbox_tiled = bbox_tiled[torch.randperm(bbox_tiled.size(0))]
    bad_bbox_shift = shift_bbox(bbox_tiled[:bad_bbox_size], truncation=True)
    bad_bbox_random = random_bbox(bad_bbox_size, max_size=1.0, truncation=True)
    bad_bbox = torch.cat([bad_bbox_shift, bad_bbox_random], 0)
    bad_bbox = bad_bbox[torch.randperm(bad_bbox.size(0))][:bad_bbox_size]
    bad_bbox_label = (
        torch.zeros([bad_bbox_size], dtype=bbox_label.dtype) + noise_label
    )

    # Create dup bbox.
    bbox_tiled = bbox_tiled[torch.randperm(bbox_tiled.size(0))]
    dup_bbox = jitter_bbox(
        bbox_tiled[:dup_bbox_size], min_range=0, max_range=0.1, truncation=True
    )
    dup_bbox_label = (
        torch.zeros([dup_bbox_size], dtype=bbox_label.dtype) + noise_label
    )

    # Jitter positive bbox.
    if max_jitter > 0:
        bbox = jitter_bbox(
            bbox, min_range=0, max_range=max_jitter, truncation=True
        )

    if torch.rand([]) < mix_rate:
        # Mix the bbox with bad bbox, appended by dup bbox.
        bbox_new = torch.cat([bbox, bad_bbox], 0)
        bbox_new_label = torch.cat([bbox_label, bad_bbox_label], 0)
        idx = torch.randperm(bbox_new.size(0))
        bbox_new = bbox_new[idx]
        bbox_new_label = bbox_new_label[idx]
        bbox_new = torch.cat([bbox_new, dup_bbox], 0)
        bbox_new_label = torch.cat([bbox_new_label, dup_bbox_label], 0)
    else:
        # Merge bad bbox and dup bbox into noise bbox.
        noise_bbox = torch.cat([bad_bbox, dup_bbox], 0)
        noise_bbox_label = torch.cat([bad_bbox_label, dup_bbox_label], 0)

        if n_noise_bbox > 0:
            idx = torch.randperm(n_noise_bbox)
            noise_bbox = noise_bbox[idx]
            noise_bbox_label = noise_bbox_label[idx]

        # Append noise bbox to bbox and create mask.
        bbox_new = torch.cat([bbox, noise_bbox], 0)
        bbox_new_label = torch.cat([bbox_label, noise_bbox_label], 0)

    return bbox_new, bbox_new_label


def inject_noise_bbox(labels, max_instances_per_image, noise_label):
    labels = copy.deepcopy(labels)
    num_instances = labels["bbox"].size(0)
    if num_instances < max_instances_per_image:
        n_noise_bbox = max_instances_per_image - num_instances
        labels["bbox"], labels["label"] = augment_bbox(
            labels["bbox"], labels["label"], 0.0, n_noise_bbox, noise_label
        )
    return labels


if __name__ == "__main__":
    test_labels = {
        "bbox": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        "label": torch.tensor([1, 2, 3]),
    }
    print(inject_noise_bbox(test_labels, 10, noise_label=81))
