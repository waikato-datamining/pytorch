import argparse
import json
import os
import torch
import traceback

from sfp import Poller
from PIL import Image
from falcontune.model import MODEL_CONFIGS
from falcontune.backend import BACKENDS
from falcontune.data import make_prompt
from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from falcontune.model.utils import model_to_half
from falcontune.generate import AMPWrapper


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


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


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
        instruction = d["instruction"] if ("instruction" in d) else ""
        input_ = d["input"] if ("input" in d) else ""
        input_data = make_prompt(instruction, input_=input_) if (len(instruction) > 0) else prompt

        input_ids = poller.params.tokenizer.encode(input_data, return_tensors="pt").to(poller.params.model.device)

        with torch.no_grad():
            generated_ids = poller.params.model.generate(
                inputs=input_ids,
                do_sample=poller.params.do_sample,
                max_new_tokens=poller.params.max_new_tokens,
                top_p=poller.params.top_p,
                top_k=poller.params.top_k,
                temperature=poller.params.temperature,
                use_cache=poller.params.use_cache,
                eos_token_id=poller.params.tokenizer.eos_token_id,
                bos_token_id=poller.params.tokenizer.bos_token_id,
                pad_token_id=poller.params.tokenizer.eos_token_id,
                num_beams=poller.params.num_beams
            )

        output = poller.params.tokenizer.decode(generated_ids.cpu().tolist()[0], skip_special_tokens=True)
        if len(instruction) > 0:
            output = format_output(output)
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
        description="Falcontune - Prediction (file-polling)",
        prog="falcon_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', choices=MODEL_CONFIGS, required=True, help='Type of model to load')
    parser.add_argument('--weights', type=str, required=True, help='Path to the base model weights.')
    parser.add_argument("--lora_apply_dir", default=None, required=False, help="Path to directory from which LoRA has to be applied before training.")
    parser.add_argument('--max_new_tokens', type=int, default=400, help='Maximum new tokens of the sequence to be generated.')
    parser.add_argument('--top_p', type=float, default=.95, help='Top p sampling parameter.')
    parser.add_argument('--top_k', type=int, default=40, help='Top k sampling parameter.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('--use_cache', action="store_true", help='Use cache when generating.')
    parser.add_argument('--do_sample', action="store_true", help='Sampling when generating.')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams.')
    parser.add_argument('--backend', type=str, default='triton', choices=BACKENDS, required=False, help='Change the default backend.')
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
        print("Loading weights: %s" % parsed.weights)
    model, tokenizer = load_model(
        parsed.model,
        parsed.weights,
        backend=parsed.backend)
    if parsed.lora_apply_dir is not None:
        model = load_adapter(model, lora_apply_dir=parsed.lora_apply_dir)
    if getattr(model, 'loaded_in_4bit', False):
        model_to_half(model)

    if parsed.verbose:
        print('Applying AMP Wrapper ...')
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()

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
    poller.params.model = model
    poller.params.tokenizer = tokenizer
    poller.params.max_new_tokens = parsed.max_new_tokens
    poller.params.top_p = parsed.top_p
    poller.params.top_k = parsed.top_k
    poller.params.temperature = parsed.temperature
    poller.params.use_cache = parsed.use_cache
    poller.params.do_sample = parsed.do_sample
    poller.params.num_beams = parsed.num_beams
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
