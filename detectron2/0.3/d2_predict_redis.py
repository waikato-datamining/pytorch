import io
import numpy as np
import torch
import traceback

from datetime import datetime
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from PIL import Image
from wai.annotations.image_utils import polygon_to_minrect, lists_to_polygon, polygon_to_bbox
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from opex import ObjectPredictions, ObjectPrediction, Polygon, BBox


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
        predictor = config.predictor
        img = read_image_bytes(msg_cont.message['data'], format="BGR")
        predictions = predictor(img)
        if "instances" not in predictions:
            raise Exception("Didn't find 'instances' in the predictions dictionary!")
        instances = predictions["instances"].to(config.cpu_device)
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
            polygons = None

        objs = []
        for i in range(num_instances):
            score = scores[i].item()
            if score >= config.score_threshold:
                label = classes[i].item()
                label_str = config.class_names[label]
                box = boxes[i].tensor.numpy()
                x0, y0, x1, y1 = box[0]

                px = None
                py = None

                if polygons is not None:
                    poly = polygons[i][0]
                    px = []
                    py = []
                    for n in range(len(poly)):
                        if n % 2 == 0:
                            px.append(poly[n])
                        else:
                            py.append(poly[n])
                    if config.fit_bbox_to_polygon:
                        if len(px) >= 3:
                            x0, y0, x1, y1 = polygon_to_bbox(lists_to_polygon(px, py))

                bbox = BBox(left=int(x0), top=int(y0), right=int(x1), bottom=int(y1))
                p = []
                for j in range(len(px)):
                    p.append([int(px[j]), int(py[j])])
                poly = Polygon(points=p)
                pred = ObjectPrediction(label=label_str, score=score, bbox=bbox, polygon=poly)
                objs.append(pred)

        preds = ObjectPredictions(id=str(start_time), timestamp=str(start_time), objects=objs)
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
    parser = create_parser('Detectron2 - Prediction (Redis)', prog="d2_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The model config file to use')
    parser.add_argument('--labels', metavar='FILE', required=True, help='the file with the labels (comma-separate list)')
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    parser.add_argument('--output_minrect', action='store_true', help='When outputting polygons whether to store the minimal rectangle around the objects in the CSV files as well', required=False, default=False)
    parser.add_argument('--fit_bbox_to_polygon', action='store_true', help='Whether to fit the bounding box to the polygon', required=False, default=False)
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    config = Container()
    config.score_threshold = parsed.score_threshold
    config.output_width_height = parsed.output_width_height
    config.output_minrect = parsed.output_minrect
    config.fit_bbox_to_polygon = parsed.fit_bbox_to_polygon
    config.verbose = parsed.verbose

    # loads labels
    print("Loading labels...")
    labels = load_labels(parsed.labels)
    num_classes = len(labels)
    if parsed.verbose:
        print("# classes:", num_classes)
        print("classes:", labels)
    config.class_names = labels

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
