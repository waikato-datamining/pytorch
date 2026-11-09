import argparse
import redis
import traceback


def test_image(image_file, channel, redis_host="localhost", redis_port=6379, redis_db=0):
    """
    Method for performing predictions on images.

    :param image_file: the image to broadcast
    :type image_file: str
    :param channel: the channel to broadcast the image on
    :type channel: str
    :param redis_host: the redis host to use
    :type redis_host: str
    :param redis_port: the port the redis host runs on
    :type redis_port: int
    :param redis_db: the redis database to use
    :type redis_db: int
    """

    # connect
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    # prepare data
    with open(image_file, "rb") as f:
        image = f.read()

    r.publish(channel, image)


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Detectron2 - Test image (Redis)',
                                     prog="d2_test_image_redis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--redis_host', metavar='HOST', required=False, default="localhost", help='The redis server to connect to')
    parser.add_argument('--redis_port', metavar='PORT', required=False, default=6379, type=int, help='The port the redis server is listening on')
    parser.add_argument('--redis_db', metavar='DB', required=False, default=0, type=int, help='The redis database to use')
    parser.add_argument('--image', metavar='FILE', required=True, default=None, help='The image to use for testing')
    parser.add_argument('--channel', metavar='ID', required=True, default=None, help='The channel to broadcast the image on')

    parsed = parser.parse_args(args=args)

    test_image(parsed.image, parsed.channel,
               redis_host=parsed.redis_host, redis_port=parsed.redis_port, redis_db=parsed.redis_db)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
