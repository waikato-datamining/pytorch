import argparse
import base64
import json
import redis
import traceback


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="SAM - Prediction test/send (Redis)",
        prog="sam_test_redis_send",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--host', metavar='HOST', required=False, default="localhost", help='The redis server to connect to')
    parser.add_argument('-p', '--port', metavar='PORT', required=False, default=6379, type=int, help='The port the redis server is listening on')
    parser.add_argument('-d', '--database', metavar='DB', required=False, default=0, type=int, help='The redis database to use')
    parser.add_argument('-c', '--channel', metavar='CHANNEL', required=True, default=None, help='The channel to broadcast the content on')
    parser.add_argument('-I', '--image', metavar="FILE", type=str, required=True, help='The JPG to send to be processed.')
    parser.add_argument('-P', '--prompt', metavar="FILE", type=str, required=True, help='The JSON file with the prompt for SAM.')
    parsed = parser.parse_args(args=args)

    r = redis.Redis(host=parsed.host, port=parsed.port, db=parsed.database)

    # read image
    with open(parsed.image, "rb") as f:
        content = f.read()

    # load prompt
    with open(parsed.prompt, "r") as f:
        prompt = json.load(f)

    # assemble and send data
    d = {
        "image": base64.encodebytes(content).decode("ascii"),
        "prompt": prompt,
    }
    r.publish(parsed.channel, json.dumps(d))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
