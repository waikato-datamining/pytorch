import argparse
import json
import os
import torch
import traceback

from sfp import Poller
from PIL import Image
from predict_common import load_gpt2xl, load_gptneo, predict_gpt2xl, predict_gptneo
from predict_common import MODEL_TYPES, MODEL_TYPE_GPT2XL, MODEL_TYPE_GPTNEO


SUPPORTED_EXTS = [".json"]
""" supported file extensions (lower case with dot). """


def check_prompt(fname, poller):
    """
    Check method that ensures the json is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    try:
        with open(fname, "r") as fp:
            json.load(fp)
        result = True
    except:
        result = False
    poller.debug("JSON complete:", fname, "->", result)
    return result


def process_prompt(fname, output_dir, poller):
    """
    Method for processing a JSON prompt.

    :param fname: the prompt to process
    :type fname: str
    :param output_dir: the directory to write the prompt to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        output_response = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".txt")

        with open(fname, "r") as fp:
            d = json.load(fp)
        prompt = d["prompt"] if ("prompt" in d) else ""

        if poller.params.model_type == MODEL_TYPE_GPT2XL:
            output = predict_gpt2xl(prompt, poller.params.model, poller.params.tokenizer,
                                    length=poller.params.length, device=poller.params.device,
                                    temperature=poller.params.temperature, top_p=poller.params.top_p,
                                    top_k=poller.params.top_k, use_cache=poller.params.use_cache,
                                    do_sample=poller.params.do_sample)
        elif poller.params.model_type == MODEL_TYPE_GPTNEO:
            output = predict_gptneo(prompt, poller.params.model, poller.params.tokenizer,
                                    length=poller.params.length, device=poller.params.device,
                                    temperature=poller.params.temperature, top_p=poller.params.top_p,
                                    top_k=poller.params.top_k, use_cache=poller.params.use_cache,
                                    do_sample=poller.params.do_sample)
        else:
            raise Exception("Unsupported model type: %s" % poller.params.model_type)

        with open(output_response, "w") as fp:
            fp.write(output)

        result.append(output_response)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process prompt: %s\n%s" % (fname, traceback.format_exc()))
    return result


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="finetune-gpt2xl - Prediction (file-polling)",
        prog="gpt_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', choices=MODEL_TYPES, required=True, help='Type of model to load')
    parser.add_argument('--model_path', type=str, metavar="DIR", required=True, help='The directory with the fine-tuned model')
    parser.add_argument('--device', type=str, required=False, default="cuda", help='The device to run the inference on, eg "cuda" or "cpu"')
    parser.add_argument('--fp16', action='store_true', help='Whether to use half-precision floating point', required=False, default=False)
    parser.add_argument('--length', type=int, default=400, help='The sequence length.')
    parser.add_argument('--top_p', type=float, default=.9, help='Top p sampling parameter.')
    parser.add_argument('--top_k', type=int, default=0, help='Top k sampling parameter.')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature.')
    parser.add_argument('--use_cache', action="store_true", help='Use cache when generating.')
    parser.add_argument('--do_sample', action="store_true", help='Sampling when generating.')
    parser.add_argument('--prediction_in', help='Path to the images to process', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the folder for the prediction files', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary folder for the prediction files', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    if parsed.verbose:
        print("Loading model: %s" % parsed.model_path)
    if parsed.model_type == MODEL_TYPE_GPT2XL:
        model, tokenizer = load_gpt2xl(parsed.model_path, device=parsed.device, fp16=parsed.fp16)
    elif parsed.model_type == MODEL_TYPE_GPTNEO:
        model, tokenizer = load_gptneo(parsed.model_path, device=parsed.device, fp16=parsed.fp16)
    else:
        raise Exception("Unsupported model type: %s" % parsed.model_type)

    poller = Poller()
    poller.input_dir = parsed.prediction_in
    poller.output_dir = parsed.prediction_out
    poller.tmp_dir = parsed.prediction_tmp
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = parsed.delete_input
    poller.verbose = parsed.verbose
    poller.progress = not parsed.quiet
    poller.check_file = check_prompt
    poller.process_file = process_prompt
    poller.poll_wait = parsed.poll_wait
    poller.continuous = parsed.continuous
    poller.use_watchdog = parsed.use_watchdog
    poller.watchdog_check_interval = parsed.watchdog_check_interval
    poller.params.model_type = parsed.model_type
    poller.params.model = model
    poller.params.tokenizer = tokenizer
    poller.params.top_p = parsed.top_p
    poller.params.top_k = parsed.top_k
    poller.params.temperature = parsed.temperature
    poller.params.use_cache = parsed.use_cache
    poller.params.do_sample = parsed.do_sample
    poller.params.device = parsed.device
    poller.params.length = parsed.length
    poller.poll()


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
