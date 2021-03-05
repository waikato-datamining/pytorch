# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 University of Waikato, Hamilton, NZ

# based on:
# https://github.com/facebookresearch/detectron2/blob/v0.3/tools/train_net.py

import argparse
import logging
import os
import torch
import traceback
import detectron2.utils.comm as comm
from collections import OrderedDict
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def train(cfg):
    """
    Trains the model with the specified configuration.

    :param cfg: the configuration to use
    :return: OrderedDict of results, if evaluation is enabled. Otherwise None.
    :rtype: OrderedDict
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def main(args=None):
    """
    Performs the model building/evaluation.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Detectron2 - COCO Training',
                                     prog="d2_train_coco",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_annotations', metavar='FILE', required=True, help='the COCO training JSON file')
    parser.add_argument('--train_images', metavar='DIR', required=True, help='the directory with the training images')
    parser.add_argument('--test_annotations', metavar='FILE', required=True, help='the COCO test JSON file')
    parser.add_argument('--test_images', metavar='DIR', required=True, help='the directory with the test images')
    parser.add_argument('--config', metavar='FILE', required=True, help='the model config file to use')
    parser.add_argument('--output_dir', metavar='DIR', required=True, help='the directory for storing the output')

    parsed = parser.parse_args(args=args)

    print(parsed)

    # load config
    print("Loading config...")
    cfg = get_cfg()
    cfg.merge_from_file(parsed.config)
    cfg.OUTPUT_DIR = parsed.output_dir
    cfg.DATASETS.TRAIN = ("coco_ext_train",)
    cfg.DATASETS.TEST = ("coco_ext_test",)
    cfg.freeze()

    # register datasets
    print("Registering datasets...")
    register_coco_instances("coco_ext_train", {}, parsed.train_annotations, parsed.train_images)
    register_coco_instances("coco_ext_test", {}, parsed.test_annotations, parsed.test_images)

    train(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
