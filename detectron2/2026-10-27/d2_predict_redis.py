import io
import numpy as np
import torch
import traceback
from PIL import Image
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from opex import ObjectPredictions, ObjectPrediction, Polygon, BBox
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log

from d2_predict_common import load_labels, labels_to_mapping
from d2_predict_slicing import predict, predict_slice, polygon_nms, detections_to_opex


def read_image_bytes(array, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        array (bytearray): image data as byte array
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray): an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    f = io.BytesIO(array)
    image = Image.open(f)
    # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
    image = _apply_exif_orientation(image)
    return convert_PIL_to_numpy(image, format)


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_images - start processing image")
        img = read_image_bytes(msg_cont.message['data'], format="BGR")
        if (config.slice_height is not None) and (config.slice_width is not None):
            detections = predict(img, config.predictor, score_threshold=config.score_threshold,
                                 slice_height=config.slice_height, slice_width=config.slice_width,
                                 overlap_height_ratio=config.overlap_height_ratio, overlap_width_ratio=config.overlap_width_ratio,
                                 merge_ios_threshold=config.merge_ios_threshold, nms_ios_threshold=config.nms_ios_threshold)
        else:
            detections = predict_slice(config.predictor, img, 0, 0, score_threshold=config.score_threshold)
            detections = polygon_nms(detections, ios_threshold=config.nms_ios_threshold)
        preds = detections_to_opex(
            detections,
            image_id=str(start_time),
            timestamp=str(start_time),
            class_mapping=config.class_mapping,
            simplify_tolerance=config.simplify_tolerance)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds.to_json_string())
        if config.verbose:
            log("process_images - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('Detectron2 - Prediction (Redis)', prog="d2_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The model config file to use')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--score_threshold', metavar="THRESH", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument('--slice_height', metavar="HEIGHT", type=int, default=None, help="The height of the slices, uses full height if None")
    parser.add_argument('--slice_width', metavar="WIDTH", type=int, default=None, help="The width of the slices, uses width height if None")
    parser.add_argument('--overlap_height_ratio', metavar="RATIO", type=float, default=0.2, help="The overlap ratio for the height to use when slicing.")
    parser.add_argument('--overlap_width_ratio', metavar="RATIO", type=float, default=0.2, help="The overlap ratio for the width to use when slicing.")
    parser.add_argument('--merge_ios_threshold', metavar="THRESH", type=float, default=0.5, help="The IoS (intersect over small) threshold for merging polygons.")
    parser.add_argument('--nms_ios_threshold', metavar="THRESH", type=float, default=0.7, help="The IoS (intersect over small) threshold for removing overlapping polygons.")
    parser.add_argument('--simplify_tolerance', metavar="TOL", type=float, default=1.0, help="The tolerance factor for simplifying the polygons, the higher the simpler.")
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    config = Container()
    config.score_threshold = parsed.score_threshold
    config.slice_height = parsed.slice_height
    config.slice_width = parsed.slice_width
    config.overlap_height_ratio = parsed.overlap_height_ratio
    config.overlap_width_ratio = parsed.overlap_width_ratio
    config.merge_ios_threshold = parsed.merge_ios_threshold
    config.nms_ios_threshold = parsed.nms_ios_threshold
    config.simplify_tolerance =parsed.simplify_tolerance
    config.verbose = parsed.verbose

    # loads labels
    print("Loading labels...")
    labels = load_labels(parsed.labels)
    num_classes = len(labels)
    if parsed.verbose:
        print("# classes:", num_classes)
        print("classes:", labels)
    config.class_names = labels
    config.class_mapping = labels_to_mapping(labels)

    # load config
    print("Loading config...")
    cfg = get_cfg()
    cfg.merge_from_file(parsed.config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = parsed.model
    config.config = cfg
    config.cpu_device = torch.device("cpu")
    config.predictor = DefaultPredictor(cfg)

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
