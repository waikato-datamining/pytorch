import argparse
import os
import traceback

from PIL import Image
from image_complete import auto
from sfp import Poller

import torch

from pic.utils import load_state, state_to_model, state_to_transforms

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
        with torch.no_grad():
            img = Image.open(fname)
            batch_t = torch.unsqueeze(poller.params.transform(img), 0)
            out = poller.params.model(batch_t)
            prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
            _, indices = torch.sort(out, descending=True)
            top = ([(poller.params.classes[idx], prob[idx].item()) for idx in indices[0][:poller.params.top_x]])
            output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".csv")
            with open(output_file, "w") as rf:
                rf.write("label,probability\n")
                for t in top:
                    rf.write("%s,%.2f\n" % (t[0], t[1]))
            result.append(output_file)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def poll(model, input_dir, output_dir, tmp_dir=None, delete_input=False, quiet=False, verbose=False,
         poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0, transform=None,
         classes=None, top_x=5):
    """
    Performs the polling and processing of the images.

    :param model: the model use
    :type model: object
    :param input_dir: the input directory to poll for images
    :type input_dir: str
    :param output_dir: the output directory to place the predictions and move the input images (when not deleting them)
    :type output_dir: str
    :param tmp_dir: the directory to write the predictions to first before moving them to the output directory
    :type tmp_dir: str
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    :param verbose: whether to be more verbose with the output
    :type verbose: bool
    :param poll_wait: the time in seconds to wait between polls (when not using watchdogs)
    :type poll_wait: float
    :param continuous: whether to process images continuously
    :type continuous: bool
    :param use_watchdog: whether to use directory watchdogs instead of time-based polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the time in seconds between watchdog check intervals
    :type watchdog_check_interval: float
    :param transform: the transformations to apply to the images
    :type transform: torchvision.transforms
    :param classes: the list of class labels to use
    :type classes: list
    :param top_x: the top_x predictions to output
    :type top_x: int
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
    poller.params.model = model
    poller.params.transform = transform
    poller.params.classes = classes
    poller.params.top_x = top_x
    poller.poll()


def main(args=None):
    """
    Performs the polling and processing of the images.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(description='PyTorch Image Classification - Poll',
                                     prog="pic-poll",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='FILE', required=True,
                        help='The model state to use')
    parser.add_argument('-i', '--prediction_in', metavar='DIR', required=True,
                        help='The input directory to poll for images to make predictions for')
    parser.add_argument('-o', '--prediction_out', metavar='DIR', required=True,
                        help='The directory to place predictions in and move input images to')
    parser.add_argument('-t', '--prediction_tmp', metavar='DIR',
                        help='The directory to place the prediction files in first before moving them to the output directory')
    parser.add_argument('--top_x', metavar='INT', type=int, default=5,
                        help='The top X categories to return')
    parser.add_argument('--poll_wait', type=float, required=False, default=1.0,
                        help='poll interval in seconds when not using watchdog mode')
    parser.add_argument('--continuous', action='store_true', required=False, default=False,
                        help='Whether to continuously load test images and perform prediction')
    parser.add_argument('--use_watchdog', action='store_true', required=False, default=False,
                        help='Whether to react to file creation events rather than performing fixed-interval polling')
    parser.add_argument('--watchdog_check_interval', type=float, required=False, default=10.0,
                        help='check interval in seconds for the watchdog')
    parser.add_argument('--delete_input', action='store_true', required=False, default=False,
                        help='Whether to delete the input images rather than move them to --prediction_out directory')
    parser.add_argument('--verbose', action='store_true', required=False, default=False,
                        help='Whether to output more logging info')
    parser.add_argument('--quiet', action='store_true', required=False, default=False,
                        help='Whether to suppress output')
    parsed = parser.parse_args(args=args)

    with torch.no_grad():
        state = load_state(parsed.model)
        model = state_to_model(state)
        transform = state_to_transforms(state)

        poll(model, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
             delete_input=parsed.delete_input, quiet=parsed.quiet, verbose=parsed.verbose,
             poll_wait=parsed.poll_wait, continuous=parsed.continuous, use_watchdog=parsed.use_watchdog,
             watchdog_check_interval=parsed.watchdog_check_interval, transform=transform,
             classes=state['classes'], top_x=parsed.top_x)


def sys_main():
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    :rtype: int
    """

    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
