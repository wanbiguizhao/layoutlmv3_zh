#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import detection_utils as utils

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer,DefaultPredictor
from PIL import Image
import numpy as np
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    """
    register publaynet first
    """
    register_coco_instances(
        "publaynet_train",
        {},
        cfg.PUBLAYNET_DATA_DIR_TRAIN + ".json",
        cfg.PUBLAYNET_DATA_DIR_TRAIN
    )

    register_coco_instances(
        "publaynet_val",
        {},
        cfg.PUBLAYNET_DATA_DIR_TEST + ".json",
        cfg.PUBLAYNET_DATA_DIR_TEST
    )

    register_coco_instances(
        "icdar2019_train",
        {},
        cfg.ICDAR_DATA_DIR_TRAIN + ".json",
        cfg.ICDAR_DATA_DIR_TRAIN
    )

    register_coco_instances(
        "icdar2019_test",
        {},
        cfg.ICDAR_DATA_DIR_TEST + ".json",
        cfg.ICDAR_DATA_DIR_TEST
    )

    image_path=args.input
    original_image=Image.open(image_path)
    image = np.asarray(original_image)

    model = DefaultPredictor(cfg)
    res=model(image)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
    res = MyTrainer.test(cfg, model)
    return res



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--input", help="image path")
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy
        #允许debug sh启动的程序了。

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
#python train_net.py --config-file cascade_layoutlmv3.yaml --eval-only --num-gpus 0 MODEL.WEIGHTS ~/ms/layoutlmv3-base-finetuned-publaynet/model_final.pth OUTPUT_DIR output PUBLAYNET_DATA_DIR_TEST /media/liukun/7764-4284/ai/publaynet/val
#python demo.py --config-file ~/ms/layoutlmv3/examples/object_detection/cascade_layoutlmv3.yaml --input /media/liukun/7764-4284/ai/publaynet/val/PMC3335537_00001.jpg  --opts MODEL.WEIGHTS ~/ms/layoutlmv3-base-finetuned-publaynet/model_final.pth 