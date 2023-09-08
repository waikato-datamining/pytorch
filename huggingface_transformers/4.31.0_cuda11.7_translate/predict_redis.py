import base64
import json
import torch
import traceback

from io import BytesIO
from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from PIL import Image
from predict_common import load_model, translate


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

        text = d["prompt"] if ("prompt" in d) else ""

        output = translate(config.tokenizer, config.model,
                           config.prompt, text, device=config.device,
                           max_new_tokens=config.max_new_tokens,
                           top_p=config.top_p, top_k=config.top_k,
                           do_sample=config.do_sample)

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
    parser = create_parser('translation - Prediction (Redis)', prog="translation_predict_redis", prefix="redis_")
    parser.add_argument('--model_path', type=str, metavar="DIR", required=True, help='The directory with the fine-tuned model')
    parser.add_argument('--device', type=str, required=False, default="cuda", help='The device to run the inference on, eg "cuda" or "cpu"')
    parser.add_argument('--fp16', action='store_true', help='Whether to use half-precision floating point', required=False, default=False)
    parser.add_argument('--prompt', type=str, required=True, default=None, help='The prompt that was used for training, e.g., "translate English to French:"')
    parser.add_argument('--max_new_tokens', type=int, default=40, help='The maximum number of tokens.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top p sampling parameter.')
    parser.add_argument('--top_k', type=int, default=30, help='Top k sampling parameter.')
    parser.add_argument('--do_sample', action="store_true", help='Sampling when generating.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    if parsed.verbose:
        print("Loading model: %s" % parsed.model_path)
    tokenizer, model = load_model(parsed.model_path, device=parsed.device, fp16=parsed.fp16)

    config = Container()
    config.model = model
    config.tokenizer = tokenizer
    config.top_p = parsed.top_p
    config.top_k = parsed.top_k
    config.do_sample = parsed.do_sample
    config.device = parsed.device
    config.max_new_tokens = parsed.max_new_tokens
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
