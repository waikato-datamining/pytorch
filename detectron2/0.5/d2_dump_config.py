# Copyright (c) 2022 University of Waikato, Hamilton, NZ

import argparse
import traceback
from detectron2.config import get_cfg


def main(args=None):
    """
    Performs the model building/evaluation.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Detectron2 - Dump configuration',
                                     prog="d2_dump_config",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_in', metavar='FILE', required=True, help='the YAML model config file to use as input')
    parser.add_argument('--num_classes', metavar='NUM', required=True, help='the number of classes in the dataset')
    parser.add_argument('--output_dir', metavar='DIR', required=True, help='the directory to store the model and logging files in')
    parser.add_argument('--config_out', metavar='FILE', required=True, help='the YAML file to store the fully expanded configuration in')

    parsed = parser.parse_args(args=args)

    # load config
    print("Loading input config: %s" % parsed.config_in)
    cfg = get_cfg()
    cfg.merge_from_file(parsed.config_in)
    cfg.OUTPUT_DIR = parsed.output_dir
    cfg.DATASETS.TRAIN = ("coco_ext_train",)
    cfg.DATASETS.TEST = ("coco_ext_test",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = parsed.num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = parsed.num_classes
    cfg.freeze()

    print("Writing full configuration to: %s" % parsed.config_out)
    with open(parsed.config_out, "w") as f:
        f.write(cfg.dump())
        f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
