import base64
import json
import traceback

from io import BytesIO
from datetime import datetime
from image_complete import auto
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
        prompt = data["prompt"]
        im_data = base64.b64decode(data["image"])
        if not auto.is_image_complete(im_data):
            log("Invalid image data!")
        im = Image.open(BytesIO(im_data))
        # predict
        mask, contours = predict(config.model, im, prompt)
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
                    "type": "sam",
                    "model_type": config.model_type,
                    "model_file": config.model_file,
                },
                "prompt": prompt
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
    parser = create_parser('SAM - Prediction (Redis)', prog="sam_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The SAM checkpoint to use.')
    parser.add_argument('--model_type', choices=["default", "vit_h", "vit_l", "vit_b"], required=False, help='The type of the checkpoint supplied.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    if parsed.verbose:
        print("Loading model/type: %s/%s" % (parsed.model, parsed.model_type))
    model_instance = load_model(parsed.model, parsed.model_type)

    config = Container()
    config.model = model_instance
    config.model_type = parsed.model_type
    config.model_file = parsed.model
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
