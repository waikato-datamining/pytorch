import cv2
import numpy as np
import torch
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, prepare_image, predict_image_opex


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

        jpg_as_np = np.frombuffer(msg_cont.message['data'], dtype=np.uint8)
        im0 = cv2.imdecode(jpg_as_np, flags=1)
        im = prepare_image(im0, config.image_size, config.model.stride, config.device)
        preds = predict_image_opex(config.model,
                                   str(start_time),
                                   im, im0,
                                   confidence_threshold=config.confidence_threshold,
                                   iou_threshold=config.iou_threshold,
                                   max_detection=config.max_detection)
        preds_str = preds.to_json_string()
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds_str)
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
    parser = create_parser('Yolov5 - Prediction (Redis)', prog="yolov5_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The ONNX Yolov5 model to use.')
    parser.add_argument('--data', metavar='FILE', type=str, required=True, help='The YAML file with the data definition (example: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml).')
    parser.add_argument('--image_size', metavar="SIZE", type=int, required=False, default=640, help='The image size to use (for width and height).')
    parser.add_argument('--confidence_threshold', metavar="0-1", type=float, required=False, default=0.25, help='The probability threshold to use for the confidence.')
    parser.add_argument('--iou_threshold', metavar="0-1", type=float, required=False, default=0.45, help='The probability threshold to use for intersect of over union (IoU).')
    parser.add_argument('--max_detection', metavar="INT", type=int, required=False, default=1000, help='The maximum number of detections.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model_instance = load_model(parsed.model, parsed.data, parsed.image_size)

    config = Container()
    config.model = model_instance
    config.confidence_threshold = parsed.confidence_threshold
    config.iou_threshold = parsed.iou_threshold
    config.max_detection = parsed.max_detection
    config.image_size = parsed.image_size
    config.device = torch.device("cpu")
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
