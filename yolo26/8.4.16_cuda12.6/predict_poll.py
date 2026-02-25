import argparse
import os
import traceback
from datetime import datetime

from PIL import Image
from image_complete import auto
from sfp import Poller

from predict_common import OUTPUT_OPEX, OUTPUT_FORMATS
from predict_common import load_model, predict_image_opex

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png"]
""" supported file extensions (lower case with dot). """


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
        output_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.suffix)

        img = Image.open(fname)

        if poller.params.output_format == OUTPUT_OPEX:
            preds = predict_image_opex(poller.params.model_params, str(start_time), img,
                                       confidence_threshold=poller.params.confidence_threshold,
                                       classes=poller.params.classes, augment=poller.params.augment,
                                       agnostic_nms=poller.params.agnostic_nms)
            preds.save_json_to_file(output_path)
            result.append(output_path)
        else:
            poller.error("Unknown output format: %s" % poller.params.output_format)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model, input_dir, output_dir, tmp_dir=None, output_format=OUTPUT_OPEX, suffix=".json",
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, confidence_threshold=0.3, classes=None, agnostic_nms=False, augment=False,
                      device="cuda", verbose=False, quiet=False):
    """
    Performs predictions on images found in input_dir and outputs the prediction PNG files in output_dir.

    :param model: the model file to use (in ONNX format)
    :param model: str
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param output_format: the output format to generate (see OUTPUT_FORMATS)
    :type output_format: str
    :param suffix: the suffix to use for the prediction files, incl extension
    :type suffix: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param confidence_threshold: the probability threshold to use (0-1)
    :type confidence_threshold: float
    :param classes: the classes to filter by (list of 0-based label indices)
    :type classes: list
    :param agnostic_nms: whether to use class-agnostic NMS
    :type agnostic_nms: bool
    :param augment: whether to use augmented inference
    :type augment: bool
    :param device: the device to run the model on, eg cuda or cuda:0
    :type device: str
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if verbose:
        print("Loading model (%s): %s" % (device, model))
    model_params = load_model(model, device=device)

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.output_format = output_format
    poller.params.suffix = suffix
    poller.params.model_params = model_params
    poller.params.confidence_threshold = confidence_threshold
    poller.params.classes = classes
    poller.params.agnostic_nms = agnostic_nms
    poller.params.augment = augment
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Yolov10 - Prediction (file-polling)",
        prog="yolov10_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The ONNX Yolov5 model to use.')
    parser.add_argument('--device', metavar="DEVICE", type=str, default="cuda", help='The device to run the model on.')
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_format', choices=OUTPUT_FORMATS, help='The type of output format to generate', default=OUTPUT_OPEX, required=False)
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default=".json", required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--confidence_threshold', metavar="0-1", type=float, required=False, default=0.25, help='The probability threshold to use for the confidence.')
    parser.add_argument('--classes', nargs='+', type=str, help='filter by class: --class person, or --class person bicycle')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    predict_on_images(parsed.model, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                      output_format=parsed.prediction_format, suffix=parsed.prediction_suffix, poll_wait=parsed.poll_wait,
                      continuous=parsed.continuous, use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                      delete_input=parsed.delete_input, confidence_threshold=parsed.confidence_threshold,
                      classes=parsed.classes, agnostic_nms=parsed.agnostic_nms, augment=parsed.augment,
                      device=parsed.device, verbose=parsed.verbose, quiet=parsed.quiet)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.
    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
