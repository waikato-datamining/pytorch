import argparse
import numpy as np
import os
import torch
import traceback
from typing import List

from datetime import datetime
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from image_complete import auto
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from PIL import Image
from sfp import Poller
from d2_predict_common import load_labels, labels_to_mapping
from d2_predict_slicing import predict, predict_slice, polygon_nms, detections_to_opex


SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

OUTPUT_OPEX = "opex"
OUTPUT_FORMATS = [OUTPUT_OPEX]


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        start_time = datetime.now()
        img = read_image(fname, format="BGR")
        if (poller.params.slice_height is not None) and (poller.params.slice_width is not None):
            detections = predict(img, poller.params.predictor, score_threshold=poller.params.score_threshold,
                                 slice_height=poller.params.slice_height, slice_width=poller.params.slice_width,
                                 overlap_height_ratio=poller.params.overlap_height_ratio, overlap_width_ratio=poller.params.overlap_width_ratio,
                                 merge_ios_threshold=poller.params.merge_ios_threshold, nms_ios_threshold=poller.params.nms_ios_threshold)
        else:
            detections = predict_slice(poller.params.predictor, img, 0, 0, score_threshold=poller.params.score_threshold)
            detections = polygon_nms(detections, ios_threshold=poller.params.nms_ios_threshold)
        output_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.suffix)
        if poller.params.output_format == OUTPUT_OPEX:
            preds = detections_to_opex(
                detections,
                image_id=str(start_time),
                timestamp=str(start_time),
                class_mapping=poller.params.class_mapping,
                simplify_tolerance=poller.params.simplify_tolerance)
            preds.save_json_to_file(output_path)
            result.append(output_path)
        else:
            poller.error("Unknown output format: %s" % poller.params.output_format)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_from_files(cfg, input_dir: str, output_dir: str, tmp_dir: str, class_names: List[str], output_format: str = OUTPUT_OPEX, suffix: str = ".json",
                       poll_wait: float = 1.0, continuous: bool = False, use_watchdog: bool = False, watchdog_check_interval: float = 10.0,
                       delete_input: bool = False, max_files: int = -1, score_threshold: float = 0.5, slice_height: int = None, slice_width: int = None,
                       overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2,
                       merge_ios_threshold: float = 0.5, nms_ios_threshold: float = 0.7, simplify_tolerance: float = 1.0,
                       verbose: bool = False, quiet: bool = False):
    """
    Method for performing predictions on images.

    :param cfg: the configuration object to use
    :type cfg:
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param class_names: labels or class names
    :type class_names: list[str]
    :param output_format: the output format to generate (see OUTPUT_FORMATS)
    :type output_format: str
    :param suffix: the suffix to use for the prediction files, incl extension
    :type suffix: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param max_files: The maximum number of files retrieve with each poll, use <0 for no restrictions.
    :type max_files: int
    :param score_threshold: the minimum score for predictions
    :type score_threshold: float
    :param slice_height: the height of the slices in pixels
    :type slice_height: int
    :param slice_width: the width of the slices in pixels
    :type slice_width: int
    :param overlap_height_ratio: the ratio for the height overlaps
    :type overlap_height_ratio: float
    :param overlap_width_ratio: the ratio for the width overlaps
    :type overlap_width_ratio: float
    :param merge_ios_threshold: the IoS threshold for merging
    :type merge_ios_threshold: float
    :param nms_ios_threshold: the IoS threshold for removing overlapping polygons
    :type nms_ios_threshold: float
    :param simplify_tolerance: the tolerance parameter for simplifying the polygons
    :type simplify_tolerance: float
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """
    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.max_files = max_files
    poller.params.config = cfg
    poller.params.labels = class_names
    poller.params.class_mapping = labels_to_mapping(class_names)
    poller.params.class_names = class_names
    poller.params.cpu_device = torch.device("cpu")
    poller.params.predictor = DefaultPredictor(cfg)
    poller.params.output_format = output_format
    poller.params.suffix = suffix
    poller.params.score_threshold = score_threshold
    poller.params.slice_height = slice_height
    poller.params.slice_width = slice_width
    poller.params.overlap_height_ratio = overlap_height_ratio
    poller.params.overlap_width_ratio = overlap_width_ratio
    poller.params.merge_ios_threshold = merge_ios_threshold
    poller.params.nms_ios_threshold = nms_ios_threshold
    poller.params.simplify_tolerance = simplify_tolerance
    poller.poll()


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Detectron2 - Prediction',
                                     prog="d2_predict",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The model config file to use')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--prediction_in', metavar='DIR', required=True, help='The input directory to poll for images to make predictions for')
    parser.add_argument('--prediction_out', metavar='DIR', required=True, help='The directory to place predictions in and move input images to')
    parser.add_argument('--prediction_tmp', metavar='DIR', help='The directory to place the prediction files in first before moving them to the output directory')
    parser.add_argument('--prediction_format', choices=OUTPUT_FORMATS, help='The type of output format to generate', default=OUTPUT_OPEX, required=False)
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default=".json", required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--max_files', type=int, default=-1, help="Maximum files to poll at a time, use -1 for unlimited", required=False)
    parser.add_argument('--score_threshold', metavar="THRESH", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument('--slice_height', metavar="HEIGHT", type=int, default=None, help="The height of the slices, uses full height if None")
    parser.add_argument('--slice_width', metavar="WIDTH", type=int, default=None, help="The width of the slices, uses width height if None")
    parser.add_argument('--overlap_height_ratio', metavar="RATIO", type=float, default=0.2, help="The overlap ratio for the height to use when slicing.")
    parser.add_argument('--overlap_width_ratio', metavar="RATIO", type=float, default=0.2, help="The overlap ratio for the width to use when slicing.")
    parser.add_argument('--merge_ios_threshold', metavar="THRESH", type=float, default=0.5, help="The IoS (intersect over small) threshold for merging polygons.")
    parser.add_argument('--nms_ios_threshold', metavar="THRESH", type=float, default=0.7, help="The IoS (intersect over small) threshold for removing overlapping polygons.")
    parser.add_argument('--simplify_tolerance', metavar="TOL", type=float, default=1.0, help="The tolerance factor for simplifying the polygons, the higher the simpler.")
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)

    parsed = parser.parse_args(args=args)

    # loads labels
    print("Loading labels...")
    labels = load_labels(parsed.labels)
    num_classes = len(labels)
    if parsed.verbose:
        print("# classes:", num_classes)
        print("classes:", labels)

    # load config
    print("Loading config...")
    cfg = get_cfg()
    cfg.merge_from_file(parsed.config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = parsed.model

    predict_from_files(cfg, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp, labels, suffix=parsed.prediction_suffix,
                       score_threshold=parsed.score_threshold, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                       use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                       delete_input=parsed.delete_input, max_files=parsed.max_files, output_format=parsed.prediction_format,
                       slice_height=parsed.slice_height, slice_width=parsed.slice_width,
                       overlap_width_ratio=parsed.overlap_width_ratio, overlap_height_ratio=parsed.overlap_height_ratio,
                       merge_ios_threshold=parsed.merge_ios_threshold, nms_ios_threshold=parsed.nms_ios_threshold,
                       simplify_tolerance=parsed.simplify_tolerance, verbose=parsed.verbose, quiet=parsed.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
