import argparse
import cv2
import os
import traceback
from image_complete import auto
from sfp import Poller
from wai.annotations.core import ImageInfo
from wai.annotations.roi.io import ROIWriter
from predict_common import load_model, prepare_image, warmup_model_if_necessary, predict_image_rois


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
        roi_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.suffix)

        img0 = cv2.imread(fname)  # BGR
        img = prepare_image(img0, poller.params.image_size, poller.params.stride, poller.params.device, poller.params.half)

        # Warmup
        b, w, h = warmup_model_if_necessary(poller.params.model, poller.params.device, img, poller.params.old_img_b,
                                            poller.params.old_img_w, poller.params.old_img_h, poller.params.augment)
        poller.params.old_img_b, poller.params.old_img_w, poller.params.old_img_h = b, w, h

        results = predict_image_rois(poller.params.model, img, img0,
                                     poller.params.old_img_w, poller.params.old_img_h,
                                     confidence_threshold=poller.params.confidence_threshold,
                                     iou_threshold=poller.params.iou_threshold, classes=poller.params.classes,
                                     augment=poller.params.augment, agnostic_nms=poller.params.agnostic_nms)

        info = ImageInfo(filename=os.path.basename(fname), size=(poller.params.old_img_w, poller.params.old_img_h))
        roiext = (info, results)
        options = ["--output=%s" % output_dir, "--no-images", "--suffix=%s" % poller.params.suffix]
        if poller.params.output_width_height:
            options.append("--size-mode")
        roiwriter = ROIWriter(options)
        roiwriter.save([roiext])
        result.append(roi_path)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model, input_dir, output_dir, tmp_dir=None, suffix="-rois.csv",
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, image_size=640, confidence_threshold=0.3, iou_threshold=0.45,
                      classes=None, agnostic_nms=False, augment=False, no_trace=False,
                      output_width_height=False, verbose=False, quiet=False):
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
    :param image_size: the image size to use for the model (applied to width and height)
    :type image_size: int
    :param confidence_threshold: the probability threshold to use (0-1)
    :type confidence_threshold: float
    :param iou_threshold: the threshold of IoU (intersect over union; 0-1)
    :type iou_threshold: float
    :param classes: the classes to filter by (list of 0-based label indices)
    :type classes: list
    :param agnostic_nms: whether to use class-agnostic NMS
    :type agnostic_nms: bool
    :param augment: whether to use augmented inference
    :type augment: bool
    :param no_trace: don't trace model
    :type no_trace: bool
    :param output_width_height: whether to output x/y/w/h instead of x0/y0/x1/y1
    :type output_width_height: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if verbose:
        print("Loading model: %s" % model)
    model_params = load_model(model, image_size, no_trace)

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
    poller.params.suffix = suffix
    poller.params.model = model_params.model
    poller.params.device = model_params.device
    poller.params.half = model_params.half
    poller.params.stride = model_params.stride
    poller.params.names = model_params.names
    poller.params.image_size = model_params.imgsz
    poller.params.old_img_w = model_params.imgsz
    poller.params.old_img_h = model_params.imgsz
    poller.params.old_img_b = 1
    poller.params.confidence_threshold = confidence_threshold
    poller.params.iou_threshold = iou_threshold
    poller.params.classes = classes
    poller.params.agnostic_nms = agnostic_nms
    poller.params.augment = augment
    poller.params.no_trace = no_trace
    poller.params.output_width_height = output_width_height
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Yolov7 - Prediction (file-polling)",
        prog="yolov7_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The ONNX Yolov5 model to use.')
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default="-rois.csv", required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--image_size', metavar="SIZE", type=int, required=False, default=640, help='The image size to use (for width and height).')
    parser.add_argument('--confidence_threshold', metavar="0-1", type=float, required=False, default=0.25, help='The probability threshold to use for the confidence.')
    parser.add_argument('--iou_threshold', metavar="0-1", type=float, required=False, default=0.45, help='The probability threshold to use for intersect of over union (IoU).')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no_trace', action='store_true', help='don`t trace model')
    parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    predict_on_images(parsed.model, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                      suffix=parsed.prediction_suffix, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                      use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                      delete_input=parsed.delete_input, image_size=parsed.image_size,
                      confidence_threshold=parsed.confidence_threshold, iou_threshold=parsed.iou_threshold,
                      classes=parsed.classes, agnostic_nms=parsed.agnostic_nms, augment=parsed.augment,
                      no_trace=parsed.no_trace, output_width_height=parsed.output_width_height,
                      verbose=parsed.verbose, quiet=parsed.quiet)


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
