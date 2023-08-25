import base64
import json
import torch
import traceback

from io import BytesIO
from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from PIL import Image
from predict_common import load_gpt2xl, load_gptneo, predict_gpt2xl, predict_gptneo
from predict_common import MODEL_TYPES, MODEL_TYPE_GPT2XL, MODEL_TYPE_GPTNEO


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
    parser = create_parser('finetune-gpt2xl  - Prediction (Redis)', prog="gpt_predict_redis", prefix="redis_")
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
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    if parsed.verbose:
        print("Loading model: %s" % parsed.model_path)
    if parsed.model_type == MODEL_TYPE_GPT2XL:
        model, tokenizer = load_gpt2xl(parsed.model_path, device=parsed.device, fp16=parsed.fp16)
    elif parsed.model_type == MODEL_TYPE_GPTNEO:
        model, tokenizer = load_gptneo(parsed.model_path, device=parsed.device, fp16=parsed.fp16)
    else:
        raise Exception("Unsupported model type: %s" % parsed.model_type)

    config = Container()
    config.model_type = parsed.model_type
    config.model = model
    config.tokenizer = tokenizer
    config.top_p = parsed.top_p
    config.top_k = parsed.top_k
    config.temperature = parsed.temperature
    config.use_cache = parsed.use_cache
    config.do_sample = parsed.do_sample
    config.device = parsed.device
    config.length = parsed.length
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
