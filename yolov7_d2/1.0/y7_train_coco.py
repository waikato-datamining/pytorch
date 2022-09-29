#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using custom coco format dataset
what you need to do is simply change the img_dir and annotation path here
Also define your own categories.
"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.modeling import build_model
from detectron2.utils import comm

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper2, MyDatasetMapper
from yolov7.utils.allreduce_norm import all_reduce_norm


def load_labels(labels_file):
    """
    Loads the labels from the specified file.
    :param labels_file: the file to load (comma-separated list)
    :type labels_file: str
    :return: the list of labels
    :rtype: list
    """
    with open(labels_file) as lf:
        line = lf.readline()
        line = line.strip()
        return line.split(",")


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
                                            mapper=MyDatasetMapper(cfg, True))

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))
        return model

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()
        if comm.get_world_size() == 1:
            self.model.update_iter(self.iter)
        else:
            self.model.module.update_iter(self.iter)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # loads labels
    print("Loading labels...")
    labels = load_labels(args.labels)
    num_classes = len(labels)
    if args.verbose:
        print("# classes:", num_classes)
        print("classes:", labels)
    
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("coco_ext_train",)
    cfg.DATASETS.TEST = ("coco_ext_test",)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.freeze()
    default_setup(cfg, args)
    register_coco_instances("coco_ext_train", {}, args.train_annotations, args.train_images)
    register_coco_instances("coco_ext_test", {}, args.test_annotations, args.test_images)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # print('trainer.start: ', trainer.start_iter)
    # trainer.model.iter = trainer.start_iter
    # print('trainer.start: ', trainer.model.iter)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--train_annotations', metavar='FILE', required=True, help='the COCO training JSON file')
    parser.add_argument('--train_images', metavar='DIR', required=True, help='the directory with the training images')
    parser.add_argument('--test_annotations', metavar='FILE', required=True, help='the COCO test JSON file')
    parser.add_argument('--test_images', metavar='DIR', required=True, help='the directory with the test images')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--output_dir', metavar='DIR', required=True, help='the directory for storing the output')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )

