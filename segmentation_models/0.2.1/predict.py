import argparse
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
import traceback

from image_complete import auto
from sfp import Poller
from common import get_preprocessing, get_augmentation, load_config


SUPPORTED_EXTS = [".jpg", ".jpeg"]
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
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_dims = image.shape
        res = poller.params.augmentation(image=image)
        res = poller.params.preprocessing(image=res['image'])
        image = res['image']
        x_tensor = torch.from_numpy(image).to(poller.params.device).unsqueeze(0)
        pr_mask = poller.params.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # multi-class?
        if len(pr_mask.shape) == 3:
            pr_mask = np.transpose(pr_mask, (1, 2, 0))
            pr_mask_gray = np.zeros((pr_mask.shape[0], pr_mask.shape[1]))
            for i in range(pr_mask.shape[2]):
                pr_mask_gray = pr_mask_gray + 1 / pr_mask.shape[2] * i * pr_mask[:, :, i]
            pr_mask = (pr_mask_gray * 255).astype(np.uint8)

        # fix size
        if (orig_dims[0] != pr_mask.shape[0]) or (orig_dims[1] != pr_mask.shape[1]):
            # undo padding:
            if pr_mask.shape[0] > orig_dims[0]:
                pad = (pr_mask.shape[0] - orig_dims[0]) // 2
                pr_mask = pr_mask[pad:orig_dims[0]+pad, 0:orig_dims[1]]
            if pr_mask.shape[1] > orig_dims[1]:
                pad = (pr_mask.shape[1] - orig_dims[1]) // 2
                pr_mask = pr_mask[0:orig_dims[0], pad:orig_dims[1]+pad]

        # not grayscale?
        if poller.params.prediction_format == "bluechannel":
            pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
            pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
            pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".png")
        cv2.imwrite(fname_out, pr_mask)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict(model, config, device, input_dir, output_dir, tmp_dir, prediction_format="grayscale",
            poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
            delete_input=False, max_files=-1, verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param model: the torch model object to use
    :param config: the configuration dictionary to use
    :type config: dict
    :param device: the device to run the inference one, like 'cuda' or 'cpu'
    :type device: str
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param prediction_format: the format to use for the prediction images (grayscale/bluechannel)
    :type prediction_format: str
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
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """
    model_params = config['model']['parameters']
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_params['encoder_name'], model_params['encoder_weights'])

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
    poller.params.model = model
    poller.params.augmentation = get_augmentation(config, 'test')
    poller.params.preprocessing = get_preprocessing(preprocessing_fn)
    poller.params.device = device
    poller.params.prediction_format = prediction_format
    poller.poll()


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Segmentation Models - Prediction',
                                     prog="sm_predict",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The configuration in JSON (.json) or YAML (.yaml, .yml) format')
    parser.add_argument('--device', metavar='DEVICE', default="cuda", help='The device to use for inference, like "cpu" or "cuda"')
    parser.add_argument('--prediction_in', metavar='DIR', required=True, help='The input directory to poll for images to make predictions for')
    parser.add_argument('--prediction_out', metavar='DIR', required=True, help='The directory to place predictions in and move input images to')
    parser.add_argument('--prediction_tmp', metavar='DIR', help='The directory to place the prediction files in first before moving them to the output directory')
    parser.add_argument('--prediction_format', metavar='FORMAT', default="grayscale", choices=["grayscale", "bluechannel"], help='The format for the prediction images')
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--max_files', type=int, default=-1, help="Maximum files to poll at a time, use -1 for unlimited", required=False)
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model = torch.load(parsed.model)

    # load config
    print("Loading config...")
    config = load_config(parsed.config)

    predict(model, config, parsed.device, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
            prediction_format=parsed.prediction_format, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
            use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
            delete_input=parsed.delete_input, max_files=parsed.max_files,
            verbose=parsed.verbose, quiet=parsed.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
