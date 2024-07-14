import io
import traceback
from datetime import datetime

import torch
from PIL import Image
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log

from predict_common import load_model, predict_image_opex


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

        img = Image.open(io.BytesIO(msg_cont.message['data']))
        preds = predict_image_opex(config.model_params, str(start_time), img,
                                   confidence_threshold=config.confidence_threshold, classes=config.classes,
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
    parser = create_parser('Yolov10 - Prediction (Redis)', prog="yolov10_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The ONNX Yolov5 model to use.')
    parser.add_argument('--device', metavar="DEVICE", type=str, default="cuda", help='The device to run the model on.')
    parser.add_argument('--confidence_threshold', metavar="0-1", type=float, required=False, default=0.25, help='The probability threshold to use for the confidence.')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model (%s): %s" % (parsed.device, parsed.model))
    model_params = load_model(parsed.model, device=parsed.device)

    config = Container()
    config.model_params = model_params
    config.confidence_threshold = parsed.confidence_threshold
    config.classes = parsed.classes
    config.agnostic_nms = parsed.agnostic_nms
    config.augment = parsed.augment
    config.device = torch.device("cpu")
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
