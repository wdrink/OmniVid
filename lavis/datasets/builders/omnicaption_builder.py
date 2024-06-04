"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from blobchunk.blobchunk import ChunkVideoTextDataset
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("omni_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ChunkVideoTextDataset
    eval_dataset_cls = ChunkVideoTextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }
