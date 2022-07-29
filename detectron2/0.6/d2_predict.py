import argparse
import numpy as np
import os
import torch
import traceback

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from image_complete import auto
from PIL import Image
from sfp import Poller
from wai.annotations.image_utils import polygon_to_minrect, lists_to_polygon, polygon_to_bbox
from wai.annotations.core import ImageInfo
from wai.annotations.roi import ROIObject
from wai.annotations.roi.io import ROIWriter


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
        predictor = poller.params.predictor
        img = read_image(fname, format="BGR")
        predictions = predictor(img)
        if not "instances" in predictions:
            raise Exception("Didn't find 'instances' in the predictions dictionary!")
        instances = predictions["instances"].to(poller.params.cpu_device)
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

        roi_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.suffix)
        img_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.mask_suffix)

        roiobjs = []
        mask_comb = None
        for i in range(num_instances):
            score = scores[i].item()
            if score >= poller.params.score_threshold:
                label = classes[i].item()
                label_str = poller.params.class_names[label]
                box = boxes[i].tensor.numpy()
                x0, y0, x1, y1 = box[0]
                x0n = x0 / image_width
                y0n = y0 / image_height
                x1n = x1 / image_width
                y1n = y1 / image_height

                px = None
                py = None
                pxn = None
                pyn = None
                bw = None
                bh = None

                if polygons is not None:
                    try:
                        poly = polygons[i][0]
                        px = []
                        py = []
                        pxn = []
                        pyn = []
                        for n in range(len(poly)):
                            if n % 2 == 0:
                                px.append(poly[n])
                                pxn.append(poly[n] / image_width)
                            else:
                                py.append(poly[n])
                                pyn.append(poly[n] / image_height)
                        if poller.params.output_minrect:
                            bw, bh = polygon_to_minrect(lists_to_polygon(px, py))
                        if poller.params.fit_bbox_to_polygon:
                            if len(px) >= 3:
                                x0, y0, x1, y1 = polygon_to_bbox(lists_to_polygon(px, py))
                                x0n, y0n, x1n, y1n = polygon_to_bbox(lists_to_polygon(pxn, pyn))
                    except:
                        poller.error("Failed to access polygon #%d: %s" % (i, traceback.format_exc()))

                if poller.params.output_mask_image:
                    if masks is not None:
                        # TODO combine masks into single image
                        pass

                roiobj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                                   poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn,
                                   minrect_w=bw, minrect_h=bh)
                roiobjs.append(roiobj)

        info = ImageInfo(filename=os.path.basename(fname), size=(image_width, image_height))
        roiext = (info, roiobjs)
        options = ["--output=%s" % output_dir, "--no-images", "--suffix=%s" % poller.params.suffix]
        if poller.params.output_width_height:
            options.append("--size-mode")
        roiwriter = ROIWriter(options)
        roiwriter.save([roiext])
        result.append(roi_path)

        if mask_comb is not None:
            im = Image.fromarray(np.uint8(mask_comb), 'P')
            im.save(img_path, "PNG")
            result.append(img_path)

    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict(cfg, input_dir, output_dir, tmp_dir, class_names, suffix="-rois.csv", mask_suffix="-mask.png",
            score_threshold=0.0, poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
            delete_input=False, max_files=-1, output_width_height=False, output_minrect=False, output_mask_image=False,
            fit_bbox_to_polygon=False, verbose=False, quiet=False):
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
    :param suffix: the suffix to use for the prediction files, incl extension
    :type suffix: str
    :param mask_suffix: the suffix to use for the mask image files, incl extension
    :type mask_suffix: str
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
    :param max_files: The maximum number of files retrieve with each poll, use <0 for no restrictions.
    :type max_files: int
    :param output_width_height: whether to output x/y/w/h instead of x0/y0/x1/y1
    :type output_width_height: bool
    :param output_minrect: when predicting polygons, whether to output the minimal rectangles around the objects as well
    :type output_minrect: bool
    :param output_mask_image: when generating masks, whether to output a combined mask image as well
    :type output_mask_image: bool
    :param fit_bbox_to_polygon: whether to fit the bounding box to the polygon
    :type fit_bbox_to_polygon: bool
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
    poller.params.class_names = class_names
    poller.params.score_threshold = score_threshold
    poller.params.output_mask_image = output_mask_image
    poller.params.output_width_height = output_width_height
    poller.params.output_minrect = output_minrect
    poller.params.fit_bbox_to_polygon = fit_bbox_to_polygon
    poller.params.cpu_device = torch.device("cpu")
    poller.params.predictor = DefaultPredictor(cfg)
    poller.params.suffix = suffix
    poller.params.mask_suffix = mask_suffix
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
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The model config file to use')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument('--prediction_in', metavar='DIR', required=True, help='The input directory to poll for images to make predictions for')
    parser.add_argument('--prediction_out', metavar='DIR', required=True, help='The directory to place predictions in and move input images to')
    parser.add_argument('--prediction_tmp', metavar='DIR', help='The directory to place the prediction files in first before moving them to the output directory')
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default="-rois.csv", required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--max_files', type=int, default=-1, help="Maximum files to poll at a time, use -1 for unlimited", required=False)
    parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    parser.add_argument('--output_minrect', action='store_true', help='When outputting polygons whether to store the minimal rectangle around the objects in the CSV files as well', required=False, default=False)
    parser.add_argument('--output_mask_image', action='store_true', help="Whether to output a mask image (PNG) when predictions generate masks", required=False, default=False)
    parser.add_argument('--mask_image_suffix', metavar='SUFFIX', help='The suffix to use for the mask images', default="-mask.png", required=False)
    parser.add_argument('--fit_bbox_to_polygon', action='store_true', help='Whether to fit the bounding box to the polygon', required=False, default=False)
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

    predict(cfg, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp, labels,
            suffix=parsed.prediction_suffix, mask_suffix=parsed.mask_image_suffix,
            score_threshold=parsed.score_threshold, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
            use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
            delete_input=parsed.delete_input, max_files=parsed.max_files,
            output_width_height=parsed.output_width_height, output_minrect=parsed.output_minrect,
            output_mask_image=parsed.output_mask_image, verbose=parsed.verbose, quiet=parsed.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
