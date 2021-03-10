import argparse
import numpy as np
import torch
import traceback
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from image_complete import auto
from sfp import Poller

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """


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


def process_image(cfg, image, labels, score_threshold=0.5):
    cpu_device = torch.device("cpu")
    predictor = DefaultPredictor(cfg)
    img = read_image(image, format="BGR")
    predictions = predictor(img)
    if not "instances" in predictions:
        raise Exception("Didn't find 'instances' in the predictions dictionary!")
    instances = predictions["instances"].to(cpu_device)
    print(instances)
    num_instances = len(instances)
    image_height, image_width = instances.image_size
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    scores = instances.scores if instances.has("scores") else None
    if instances.has("pred_masks"):
        masks = np.asarray(instances.pred_masks)
        masks = [GenericMask(x, image_height, image_width) for x in masks]
        polygons = [x.polygons for x in masks]
    else:
        masks = None
        polygons = None
    print("index score box class polygon")
    for i in range(num_instances):
        score = scores[i].item()
        if score >= score_threshold:
            cls = classes[i].item()
            cls_str = labels[cls]
            if polygons is None:
                poly = ""
            else:
                poly = polygons[i]
            print(i, score, boxes[i], cls_str, poly)
    #print(predictions["instances"].to(cpu_device))


def predict(cfg, input_dir, output_dir, tmp_dir, class_names, score_threshold=0.0,
            poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
            delete_input=False, output_width_height=False, output_mask_image=False,
            verbose=False, quiet=False):
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
    :param score_threshold: the minimum score predictions have to have
    :type score_threshold: float
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
    :param output_width_height: whether to output x/y/w/h instead of x0/y0/x1/y1
    :type output_width_height: bool
    :param output_mask_image: when generating masks, whether to output a combined mask image as well
    :type output_mask_image: bool
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
    poller.params.config = cfg
    poller.params.class_names = class_names
    poller.params.score_threshold = score_threshold
    poller.params.output_mask_image = output_mask_image
    poller.params.output_width_height = output_width_height
    poller.poll()


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
    parser.add_argument('-m', '--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('-c', '--config', metavar='FILE', required=True, help='The model config file to use')
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    # parser.add_argument('-i', '--prediction_in', metavar='DIR', required=True, help='The input directory to poll for images to make predictions for')
    # parser.add_argument('-o', '--prediction_out', metavar='DIR', required=True, help='The directory to place predictions in and move input images to')
    # parser.add_argument('-t', '--prediction_tmp', metavar='DIR', help='The directory to place the prediction files in first before moving them to the output directory')
    # parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    # parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    # parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    # parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    # parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    # parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    # parser.add_argument('--output_mask_image', action='store_true', help="Whether to output a mask image (PNG) when predictions generate masks", required=False, default=False)
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    # parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    # TODO only for testing
    parser.add_argument('--image', metavar='FILE',
                        help='The image to process')

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

    # TODO switch to predict
    process_image(cfg, parsed.image, labels, score_threshold=parsed.score_threshold)
    # predict(cfg, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp, labels,
    #         score_threshold=parsed.score_threshold, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
    #         use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
    #         delete_input=parsed.delete_input, output_width_height=parsed.output_width_height,
    #         output_mask_image=parsed.output_mask_image, verbose=parsed.verbose, quiet=parsed.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
