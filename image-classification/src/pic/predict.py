import argparse
import os
import traceback

from PIL import Image

import torch

from pic.utils import load_state, state_to_model, state_to_transforms


def main(args=None):
    """
    Performs the single image prediction.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(description='PyTorch Image Classification - Prediction',
                                     prog="pic-predict",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='FILE', required=True,
                        help='The model state to use')
    parser.add_argument('-i', '--image', metavar='FILE', required=True,
                        help='The image to apply the model to')
    parser.add_argument('--top_x', metavar='INT', type=int, default=5,
                        help='The top X categories to return')
    parsed = parser.parse_args(args=args)

    with torch.no_grad():
        state = load_state(parsed.model)
        model = state_to_model(state)
        transform = state_to_transforms(state)
        classes = state['classes']
        # in case network couldn't be fine-tuned
        while len(classes) < state['num_network_classes']:
            classes.append("dummy-%d" % len(classes))

        print("Making predictions...")
        img = Image.open(parsed.image)
        batch_t = torch.unsqueeze(transform(img), 0)
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        top = ([(classes[idx], prob[idx].item()) for idx in indices[0][:parsed.top_x]])
        print("Image: %s" % os.path.basename(parsed.image))
        print("Predictions:")
        for t in top:
            print("- %s: %.2f" % (t[0], t[1]))


def sys_main():
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    :rtype: int
    """

    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
