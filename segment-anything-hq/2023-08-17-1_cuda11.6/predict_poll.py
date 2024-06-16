import argparse
import json
import os
import traceback
from image_complete import auto
from sfp import Poller
from predict_common import load_model, predict, contours_to_opex, contours_to_list
from PIL import Image


SUPPORTED_EXTS = [".jpg", ".jpeg"]
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
        input_prompt = os.path.splitext(fname)[0] + ".prompt"
        with open(input_prompt, "r") as fp:
            prompt = json.load(fp)
        output_mask = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".png")
        output_opex = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".json")
        im = Image.open(fname)
        mask, contours = predict(poller.params.model, im, prompt)
        # save mask image
        im_out = Image.fromarray(mask)
        im_out.save(output_mask)
        result.append(output_mask)
        # save contours as opex json
        opex = contours_to_opex(contours_to_list(contours), id=os.path.splitext(os.path.basename(fname))[0])
        opex.meta = {
            "segmenter": {
                "type": "sam-hq",
                "model_type": poller.params.model_type,
                "model_file": poller.params.model_file,
            },
            "prompt": prompt
        }
        opex.save_json_to_file(output_opex)
        result.append(output_opex)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model, model_type, device, input_dir, output_dir, tmp_dir=None,
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, verbose=False, quiet=False):
    """
    Performs predictions on images found in input_dir and outputs the prediction PNG files in output_dir.

    :param model: the checkpoint file to use
    :param model: str
    :param model_type: the type of model the checkpoint represents (default|vit_h|vit_l|vit_b|vit_tiny)
    :type model_type: str
    :param device: the device to run the model on, e.g., 'cpu' or 'cuda:0'
    :type device: str
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
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
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """
    if verbose:
        print("Loading model/type: %s/%s" % (model, model_type))
    model_instance = load_model(model, model_type, device)

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
    poller.other_input_files = ["{NAME}.prompt"]
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model_instance
    poller.params.model_type = model_type
    poller.params.model_file = model
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="SAM - Prediction (file-polling)",
        prog="sam_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The SAM checkpoint to use.')
    parser.add_argument('--model_type', choices=["default", "vit_h", "vit_l", "vit_b", "vit_tiny"], required=False, help='The type of the checkpoint supplied.')
    parser.add_argument('--device', metavar="DEVICE", type=str, default="cuda:0", help='The device to run the model on.')
    parser.add_argument('--prediction_in', help='Path to the images to process', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the folder for the prediction files', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary folder for the prediction files', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    predict_on_images(parsed.model, parsed.model_type, parsed.device, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                      poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                      use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                      delete_input=parsed.delete_input, verbose=parsed.verbose,
                      quiet=parsed.quiet)


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
