import io
import traceback
from datetime import datetime

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log

from predict_common import load_model, make_features, load_label, predict, PREDICTION_FORMATS, PREDICTION_FORMAT_TEXT, PREDICTION_FORMAT_JSON


def process_audio(msg_cont):
    """
    Processes the message container, loading the audio from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_audio - start processing audio")
            
        wav_data = io.BytesIO(msg_cont.message['data'])
        preds_str = predict(config.model, wav_data, config.class_labels, top_k=config.top_k, prediction_format=config.prediction_format)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds_str)
        if config.verbose:
            log("process_audio - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_audio - finished processing audio: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_audio - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('ast - Prediction (Redis)', prog="ast_predict_redis", prefix="redis_")
    parser.add_argument('--model', type=str, help='The pretrained model to use.', required=True, default=None)
    parser.add_argument('--device', type=str, help='The device to run the model on.', required=False, default="cuda:0")
    parser.add_argument('--init_wav', type=str, help='The .wav file to use for initializing the model.', required=True, default=None)
    parser.add_argument('--class_labels', type=str, help='The CSV file with the class labels.', required=True, default=None)
    parser.add_argument('--top_k', type=int, help='The top K classes to output.', required=False, default=10)
    parser.add_argument('--prediction_format', choices=PREDICTION_FORMATS, help='The format to use for the predictions', required=False, default=PREDICTION_FORMAT_TEXT)
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading wav for initializing: %s" % parsed.init_wav)
    feats = make_features(parsed.init_wav, mel_bins=128)
    print("Loading model: %s" % parsed.model)
    model = load_model(parsed.model, parsed.device, feats.shape[0])

    config = Container()
    config.model = model
    config.feats = feats
    config.class_labels = load_label(parsed.class_labels)
    config.prediction_format = parsed.prediction_format
    config.top_k = parsed.top_k
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_audio)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
