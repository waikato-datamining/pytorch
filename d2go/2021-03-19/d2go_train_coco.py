#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (C) 2021 University of Waikato, Hamilton, NZ

"""
Detection Training Script.
"""

import logging

import detectron2.utils.comm as comm
from d2go.distributed import launch
from d2go.setup import (
    basic_argument_parser,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from d2go.utils.misc import print_metrics_table, dump_trained_model_configs
from torch.nn.parallel import DistributedDataParallel


logger = logging.getLogger("d2go.tools.train_net")


def train(cfg, output_dir, runner=None):
    """
    Performs the training.

    :param cfg: the configuration object to use
    :param output_dir: the output directory
    :param runner: the runner instance
    :return: dictionary of metrics
    """
    setup_after_launch(cfg, output_dir, runner)

    model = runner.build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=None if cfg.MODEL.DEVICE == "cpu" else [comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS,
        )

    trained_cfgs = runner.do_train(cfg, model, resume=False)
    metrics = runner.do_test(cfg, model)
    print_metrics_table(metrics)

    # dump config files for trained models
    trained_model_configs = dump_trained_model_configs(cfg.OUTPUT_DIR, trained_cfgs)
    return {
        # for e2e_workflow
        "accuracy": metrics,
        # for unit_workflow
        "model_configs": trained_model_configs,
        "metrics": metrics,
    }


def run_with_cmdline_args(args):
    """
    Launches the training process.

    :param args: the parsed command-line arguments
    """

    # loads labels
    print("Loading labels...")
    labels = load_labels(args.labels)

    cfg, output_dir, runner = prepare_for_launch(args)
    cfg.defrost()

    cfg.OUTPUT_DIR = args.output_dir
    cfg.DATASETS.TRAIN = ("coco_ext_train",)
    cfg.DATASETS.TEST = ("coco_ext_test",)
    cfg.D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES = [args.train_annotations, args.test_annotations]
    cfg.D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS = [args.train_images, args.test_images]
    cfg.D2GO_DATA.DATASETS.COCO_INJECTION.NAMES = ["coco_ext_train", "coco_ext_test"]
    cfg.D2GO_DATA.DATASETS.TRAIN_CATEGORIES = tuple(labels)
    cfg.D2GO_DATA.DATASETS.TEST_CATEGORIES = tuple(labels)
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1

    launch(
        post_mortem_if_fail_for_main(train),
        num_processes_per_machine=args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend=args.dist_backend,
        args=(cfg, output_dir, runner),
    )


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


def cli():
    parser = basic_argument_parser(requires_output_dir=False)
    parser.add_argument('--train_annotations', metavar='FILE', required=True, help='the COCO training JSON file')
    parser.add_argument('--train_images', metavar='DIR', required=True, help='the directory with the training images')
    parser.add_argument('--test_annotations', metavar='FILE', required=True, help='the COCO test JSON file')
    parser.add_argument('--test_images', metavar='DIR', required=True, help='the directory with the test images')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    run_with_cmdline_args(parser.parse_args())


if __name__ == "__main__":
    cli()
