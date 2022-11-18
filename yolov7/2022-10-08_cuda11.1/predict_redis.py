import cv2
import numpy as np
import torch
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, prepare_image, predict_image_opex, warmup_model_if_necessary


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

        img = prepare_image(im0, config.image_size, config.stride, config.device, config.half)

        # Warmup
        b, w, h = warmup_model_if_necessary(config.model, config.device, img, config.old_img_b,
                                            config.old_img_w, config.old_img_h, config.augment)
        config.old_img_b, config.old_img_w, config.old_img_h = b, w, h

        preds = predict_image_opex(config.model, str(start_time), img, im0,
                                   confidence_threshold=config.confidence_threshold,
                                   iou_threshold=config.iou_threshold, classes=config.classes,
                                   augment=config.augment, agnostic_nms=config.agnostic_nms)
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


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('Yolov7 - Prediction (Redis)', prog="yolov7_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The ONNX Yolov5 model to use.')
    parser.add_argument('--device', metavar="DEVICE", type=str, default="cuda:0", help='The device to run the model on.')
    parser.add_argument('--image_size', metavar="SIZE", type=int, required=False, default=640, help='The image size to use (for width and height).')
    parser.add_argument('--confidence_threshold', metavar="0-1", type=float, required=False, default=0.25, help='The probability threshold to use for the confidence.')
    parser.add_argument('--iou_threshold', metavar="0-1", type=float, required=False, default=0.45, help='The probability threshold to use for intersect of over union (IoU).')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no_trace', action='store_true', help='don`t trace model')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model_params = load_model(parsed.model, parsed.image_size, parsed.no_trace, device_id=parsed.device)

    config = Container()
    config.model = model_params.model
    config.device = model_params.device
    config.half = model_params.half
    config.stride = model_params.stride
    config.names = model_params.names
    config.image_size = model_params.imgsz
    config.old_img_w = model_params.imgsz
    config.old_img_h = model_params.imgsz
    config.old_img_b = 1
    config.confidence_threshold = parsed.confidence_threshold
    config.iou_threshold = parsed.iou_threshold
    config.classes = parsed.classes
    config.agnostic_nms = parsed.agnostic_nms
    config.augment = parsed.augment
    config.no_trace = parsed.no_trace
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
