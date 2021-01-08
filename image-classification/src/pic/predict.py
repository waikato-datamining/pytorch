import argparse
import traceback

from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from pic.main import NORMALIZE


def main(args=None):
    """
    Performs the single image prediction.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Image Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='FILE',
                        help='The model state to use')
    parser.add_argument('-i', '--image', metavar='FILE',
                        help='The image to apply the model to')
    parsed = parser.parse_args(args=args)

    with torch.no_grad():
        print("Loading state...")
        params = torch.load(parsed.model)
        model = models.__dict__[params['arch']]()
        model.load_state_dict(params['state_dict'])
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        classes = params['classes']
        width = params['width']
        height = params['height']

        print("Making prediction...")
        transform = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            NORMALIZE,
        ])
        img = Image.open(parsed.image)
        batch_t = torch.unsqueeze(transform(img), 0)
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        print([(classes[idx], prob[idx].item()) for idx in indices[0][:5]])


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
