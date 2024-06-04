"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np

import lavis.tasks as tasks
import torch
import torch.backends.cudnn as cudnn
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

from lavis.datasets.datasets.omnicaption_dataset import create_dataset
from lavis.datasets.datasets.omnicaption_clip_dataset import create_clip_dataset
from lavis.datasets.datasets.omnisot_dataset import create_sot_dataset
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg-path", required=True, help="path to configuration file."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    task = tasks.setup_task(cfg)
    online_mode = cfg.model_cfg.get("online_mode", False)

    if "sot" in cfg.run_cfg.evaluate_type:
        datasets = create_sot_dataset(cfg)
    elif online_mode:
        datasets = create_clip_dataset(cfg)
    else:
        datasets = create_dataset(cfg)

    if hasattr(datasets["train"][0], "tokenize"):
        vocab_size = len(datasets["train"][0].tokenize.tokenizer)
        tokenizer = datasets["train"][0].tokenize

    else:
        vocab_size = 0
        tokenizer = None

    # print("VOCAB SIZE: ", vocab_size)
    cfg.model_cfg.vocab_size = vocab_size
    model = task.build_model(cfg)
    model.set_tokenizer(tokenizer)

    runner = RunnerBase(
        cfg=cfg,
        task=task,
        model=model,
        datasets=datasets,
        auto_resume=False,
    )
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
