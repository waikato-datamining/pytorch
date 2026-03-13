import argparse
import os
import traceback

from sfp import Poller

from predict_common import load_model, make_features, load_label, predict, PREDICTION_FORMATS, PREDICTION_FORMAT_TEXT, PREDICTION_FORMAT_JSON

SUPPORTED_EXTS = [".wav"]
""" supported file extensions (lower case with dot). """


def process_audio(fname, output_dir, poller):
    """
    Method for processing audio data.

    :param fname: the audio file to process
    :type fname: str
    :param output_dir: the directory to write the audio data to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        if poller.params.prediction_format == PREDICTION_FORMAT_JSON:
            suffix = ".json"
        else:
            suffix = ".txt"
        output_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], suffix)
        preds = predict(poller.params.model, fname, poller.params.class_labels, top_k=poller.params.top_k, prediction_format=poller.params.prediction_format)
        with open(output_path, "w") as fp:
            fp.write(preds)
        result.append(output_path)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process audio file: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_audio(model, device, init_wav, class_labels, input_dir, output_dir, tmp_dir=None,
                     prediction_format=PREDICTION_FORMAT_TEXT, top_k=10,
                     poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                     delete_input=False, verbose=False, quiet=False):
    """
    Performs predictions on images found in input_dir and outputs the prediction PNG files in output_dir.

    :param model: the path to the pretrained model to use
    :type model: str
    :param device: the device to run the model on
    :type device: str
    :param init_wav: the WAV file to initialize with
    :type init_wav: str
    :param class_labels: the CSV file with the labels to load
    :type class_labels: str
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param prediction_format: the output format to use
    :type prediction_format: str
    :param top_k: the top K labels to output
    :type top_k: int
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if verbose:
        print("Loading model...")
    print("Loading wav for initializing: %s" % init_wav)
    feats = make_features(init_wav, mel_bins=128)
    print("Loading model: %s" % model)
    model = load_model(model, device, feats.shape[0])

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.process_file = process_audio
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model
    poller.params.feats = feats
    poller.params.class_labels = load_label(class_labels)
    poller.params.prediction_format = prediction_format
    poller.params.top_k = top_k
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="ast - Prediction (file-polling)",
        prog="ast_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='The pretrained model to use.', required=True, default=None)
    parser.add_argument('--device', type=str, help='The device to run the model on.', required=False, default="cuda:0")
    parser.add_argument('--init_wav', type=str, help='The .wav file to use for initializing the model.', required=True, default=None)
    parser.add_argument('--class_labels', type=str, help='The CSV file with the class labels.', required=True, default=None)
    parser.add_argument('--top_k', type=int, help='The top K classes to output.', required=False, default=10)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_format', choices=PREDICTION_FORMATS, help='The format to use for the predictions', required=False, default=PREDICTION_FORMAT_TEXT)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    predict_on_audio(parsed.model, parsed.device, parsed.init_wav, parsed.class_labels,
                     parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                     prediction_format=parsed.prediction_format, top_k=parsed.top_k,
                     poll_wait=parsed.poll_wait, continuous=parsed.continuous, use_watchdog=parsed.use_watchdog,
                     watchdog_check_interval=parsed.watchdog_check_interval, delete_input=parsed.delete_input,
                     verbose=parsed.verbose, quiet=parsed.quiet)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.
    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
