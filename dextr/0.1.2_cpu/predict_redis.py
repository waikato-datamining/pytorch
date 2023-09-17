import base64
import json
import traceback

from io import BytesIO
from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, predict, contours_to_list
from PIL import Image


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
        # read data
        data = json.loads(msg_cont.message['data'].decode())
        points = data["points"]
        label = data.get("label", 1)
        jpg_data = base64.b64decode(data["image"])
        im = Image.open(BytesIO(jpg_data))
        # predict
        mask, contours = predict(config.model, im, points, threshold=config.threshold, label=label)
        # generate output
        contours_list = contours_to_list(contours)
        im_out = Image.fromarray(mask)
        im_out_buf = BytesIO()
        im_out.save(im_out_buf, format='PNG')
        im_out_b64 = base64.encodebytes(im_out_buf.getvalue())
        output_data = {
            "contours": contours_list,
            "mask": im_out_b64.decode("ascii"),
            "meta": {
                "segmenter": {
                    "type": "dextr",
                },
                "prompt": {
                    "points": p,
                    "label": label
                }
            }
        }
        preds_str = json.dumps(output_data)
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
    parser = create_parser('DEXTR - Prediction (Redis)', prog="dextr_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=False, help='The DEXTR model to use, downloads/uses pretrained resunet101 model if omitted.')
    parser.add_argument('--threshold', metavar="0-1", type=float, required=False, default=0.5, help='The probability threshold to use for the mask and contours.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model_instance = load_model(parsed.model, "cpu")

    config = Container()
    config.model = model_instance
    config.threshold = parsed.threshold
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
