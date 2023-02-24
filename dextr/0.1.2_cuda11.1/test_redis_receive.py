import argparse
import base64
import json
import os
import redis
import traceback

from datetime import datetime
from io import BytesIO
from PIL import Image
from predict_common import contours_to_opex


DATETIME_FORMAT_URL = "https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes"


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="DEXTR - Prediction test/receive (Redis)",
        prog="dextr_test_redis_recv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--host', metavar='HOST', required=False, default="localhost", help='The redis server to connect to')
    parser.add_argument('-p', '--port', metavar='PORT', required=False, default=6379, type=int, help='The port the redis server is listening on')
    parser.add_argument('-d', '--database', metavar='DB', required=False, default=0, type=int, help='The redis database to use')
    parser.add_argument('-c', '--channel', metavar='CHANNEL', required=True, default=None, help='The channel to broadcast the content on')
    parser.add_argument('-o', '--output_dir', metavar="DIR", type=str, required=True, help='The directory to store the received data in.')
    parser.add_argument("-f", "--file_prefix", metavar="FORMAT", help="the format to use for the output files (prefix, not ext), see: %s" % DATETIME_FORMAT_URL, required=False, default="%Y%m%d_%H%M%S.%f")
    parsed = parser.parse_args(args=args)

    r = redis.Redis(host=parsed.host, port=parsed.port, db=parsed.database)

    # handler for listening/outputting
    def anon_handler(message):
        print("\nData received...")
        d = json.loads(message['data'].decode())
        bname = datetime.now().strftime(parsed.file_prefix)
        # mask
        pname = os.path.join(parsed.output_dir, bname + ".png")
        png_data = base64.decodebytes(d["mask"].encode())
        im = Image.open(BytesIO(png_data))
        im.save(pname)
        print(pname)
        # contours
        jname = os.path.join(parsed.output_dir, bname + ".json")
        opex = contours_to_opex(d["contours"], id=bname)
        opex.save_json_to_file(jname, indent=2)
        print(jname)

    # subscribe and start listening
    p = r.pubsub()
    p.psubscribe(**{parsed.channel: anon_handler})
    p.run_in_thread(sleep_time=0.001)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
