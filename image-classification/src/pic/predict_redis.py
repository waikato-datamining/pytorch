import io
import json
import torch
import traceback

from datetime import datetime
from collections import OrderedDict
from PIL import Image

from pic.utils import load_state, state_to_model, state_to_transforms
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the object detection predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        img = Image.open(io.BytesIO(msg_cont.message['data']))
        batch_t = torch.unsqueeze(config.transform(img), 0)
        out = config.model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        top = ([(config.classes[idx], prob[idx].item()) for idx in indices[0][:config.top_x]])
        predictions = OrderedDict()
        for t in top:
            predictions[str(t[0])] = float(t[1])

        msg_cont.params.redis.publish(msg_cont.params.channel_out, json.dumps(predictions))

        if config.verbose:
            log("process_images - predicted image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


def main(parsed=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param parsed: the commandline arguments, uses sys.argv if not supplied
    :type parsed: list
    """
    parser = create_parser("Uses a tflite image classification model to make predictions on images received via a Redis channel and broadcasts the predictions via another Redis channel.",
                           prog="pic-predict-redis", prefix="redis_")
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument("--top_x", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument("--verbose", action="store_true", help="whether to output some debugging information")
    parsed = parser.parse_args(args=parsed)

    with torch.no_grad():
        state = load_state(parsed.model)
        model = state_to_model(state)
        transform = state_to_transforms(state)
        classes = state['classes']
        # in case network couldn't be fine-tuned
        while len(classes) < state['num_network_classes']:
            classes.append("dummy-%d" % len(classes))

        config = Container()
        config.model = model
        config.transform = transform
        config.classes = classes
        config.top_x = parsed.top_x
        config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
