import base64
import json
import torch
import traceback

from io import BytesIO
from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from PIL import Image
from falcontune.model import MODEL_CONFIGS
from falcontune.backend import BACKENDS
from falcontune.data import make_prompt
from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from falcontune.model.utils import model_to_half
from falcontune.generate import AMPWrapper


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


def process_prompt(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_prompts - start processing prompt")
        # read data
        d = json.loads(msg_cont.message['data'].decode())

        prompt = d["prompt"] if ("prompt" in d) else ""
        instruction = d["instruction"] if ("instruction" in d) else ""
        input_ = d["input"] if ("input" in d) else ""
        input_data = make_prompt(instruction, input_=input_) if (len(instruction) > 0) else prompt

        input_ids = config.tokenizer.encode(input_data, return_tensors="pt").to(config.model.device)

        with torch.no_grad():
            generated_ids = config.model.generate(
                inputs=input_ids,
                do_sample=config.do_sample,
                max_new_tokens=config.max_new_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                temperature=config.temperature,
                use_cache=config.use_cache,
                eos_token_id=config.tokenizer.eos_token_id,
                bos_token_id=config.tokenizer.bos_token_id,
                pad_token_id=config.tokenizer.eos_token_id,
                num_beams=config.num_beams
            )

        output = config.tokenizer.decode(generated_ids.cpu().tolist()[0], skip_special_tokens=True)
        if len(instruction) > 0:
            output = format_output(output)

        msg_cont.params.redis.publish(msg_cont.params.channel_out, output)
        if config.verbose:
            log("process_prompts - response string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_prompts - finished processing prompt: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_prompts - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('Falcontune - Prediction (Redis)', prog="falcon_predict_redis", prefix="redis_")
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
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

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

    config = Container()
    config.model = model
    config.tokenizer = tokenizer
    config.max_new_tokens = parsed.max_new_tokens
    config.top_p = parsed.top_p
    config.top_k = parsed.top_k
    config.temperature = parsed.temperature
    config.use_cache = parsed.use_cache
    config.do_sample = parsed.do_sample
    config.num_beams = parsed.num_beams
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
