import cv2
import io
import numpy as np
import segmentation_models_pytorch as smp
import torch
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from common import get_preprocessing, get_augmentation, load_config


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

        array = np.frombuffer(io.BytesIO(msg_cont.message['data']).getvalue(), np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_dims = image.shape
        res = config.augmentation(image=image)
        res = config.preprocessing(image=res['image'])
        image = res['image']
        x_tensor = torch.from_numpy(image).to(config.device).unsqueeze(0)
        pr_mask = config.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # multi-class?
        if len(pr_mask.shape) == 3:
            pr_mask = np.transpose(pr_mask, (1, 2, 0))
            pr_mask_gray = np.zeros((pr_mask.shape[0], pr_mask.shape[1]))
            for i in range(pr_mask.shape[2]):
                pr_mask_gray = pr_mask_gray + 1 / pr_mask.shape[2] * i * pr_mask[:, :, i]
            pr_mask = (pr_mask_gray * 255).astype(np.uint8)

        # fix size
        if (orig_dims[0] != pr_mask.shape[0]) or (orig_dims[1] != pr_mask.shape[1]):
            # undo padding:
            if pr_mask.shape[0] > orig_dims[0]:
                pad = (pr_mask.shape[0] - orig_dims[0]) // 2
                pr_mask = pr_mask[pad:orig_dims[0]+pad, 0:orig_dims[1]]
            if pr_mask.shape[1] > orig_dims[1]:
                pad = (pr_mask.shape[1] - orig_dims[1]) // 2
                pr_mask = pr_mask[0:orig_dims[0], pad:orig_dims[1]+pad]

        # not grayscale?
        if config.prediction_format == "bluechannel":
            pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
            pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
            pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])

        out_data = cv2.imencode('.png', pr_mask)[1].tostring()
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
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
    parser = create_parser('Segmentation Models - Prediction (Redis)', prog="sm_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar='FILE', required=True, help='The model state to use')
    parser.add_argument('--config', metavar='FILE', required=True, help='The configuration in JSON (.json) or YAML (.yaml, .yml) format')
    parser.add_argument('--device', metavar='DEVICE', default="cuda", help='The device to use for inference, like "cpu" or "cuda"')
    parser.add_argument('--prediction_format', metavar='FORMAT', default="grayscale", choices=["grayscale", "bluechannel"], help='The format for the prediction images')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model = torch.load(parsed.model)

    # load config
    print("Loading config...")
    config = load_config(parsed.config)

    model_params = config['model']['parameters']
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_params['encoder_name'], model_params['encoder_weights'])

    cont = Container()
    cont.model = model
    cont.augmentation = get_augmentation(config, 'test')
    cont.preprocessing = get_preprocessing(preprocessing_fn)
    cont.device = parsed.device
    cont.prediction_format = parsed.prediction_format
    cont.verbose = parsed.verbose

    params = configure_redis(parsed, config=cont)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
