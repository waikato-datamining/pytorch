import argparse
import os
import traceback

import torch

from pic.utils import load_state, state_to_model, state_to_dims


def export(model, output_dir, output_name="model.pt"):
    """
    Performs the export of a model to TorchScript.

    :param model: the model file to export
    :type model: str
    :param output_dir: the output directory to place the predictions and move the input images (when not deleting them)
    :type output_dir: str
    :param output_name: the name of the exported file (not path)
    :type output_name: str
    """

    state = load_state(model)
    width, height = state_to_dims(state)
    model = state_to_model(state)
    model.eval()

    input_tensor = torch.rand(1, 3, width, height) # dummy tensor for 3 channels
    script_model = torch.jit.trace(model, input_tensor)
    output_file = os.path.join(output_dir, output_name)
    script_model.save(output_file)
    print("Exported: %s" % output_file)


def main(args=None):
    """
    Performs the export of a model to TorchScript.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(description='PyTorch Image Classification - Export',
                                     prog="pic-export",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='FILE', required=True,
                        help='The model state to use')
    parser.add_argument('-o', '--output_dir', metavar='DIR', required=True,
                        help='The directory to store the exported model in')
    parser.add_argument('-n', '--output_name', metavar='NAME', default="model.pt",
                        help='The name of the model file')
    parsed = parser.parse_args(args=args)

    export(parsed.model, parsed.output_dir, parsed.output_name)


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
