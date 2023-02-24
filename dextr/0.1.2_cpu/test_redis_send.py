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
        description="DEXTR - Prediction test/send (Redis)",
        prog="dextr_test_redis_send",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--host', metavar='HOST', required=False, default="localhost", help='The redis server to connect to')
    parser.add_argument('-p', '--port', metavar='PORT', required=False, default=6379, type=int, help='The port the redis server is listening on')
    parser.add_argument('-d', '--database', metavar='DB', required=False, default=0, type=int, help='The redis database to use')
    parser.add_argument('-c', '--channel', metavar='CHANNEL', required=True, default=None, help='The channel to broadcast the content on')
    parser.add_argument('-I', '--image', metavar="FILE", type=str, required=True, help='The JPG to send to be processed.')
    parser.add_argument('-P', '--point', nargs=4, type=str, required=True, help='The four extreme points (x,y) to use.')
    parsed = parser.parse_args(args=args)

    r = redis.Redis(host=parsed.host, port=parsed.port, db=parsed.database)

    # read image
    with open(parsed.image, "rb") as f:
        content = f.read()

    # parse points
    points = []
    for p in parsed.point:
        points.append([int(x) for x in p.split(",")])

    # assemble and send data
    d = {
        "image": base64.encodebytes(content).decode("ascii"),
        "points": points,
    }
    r.publish(parsed.channel, json.dumps(d))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
