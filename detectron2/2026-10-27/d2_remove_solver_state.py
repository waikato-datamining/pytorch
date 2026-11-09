# based on:
# https://github.com/tianzhi0549/FCOS/blob/master/tools/remove_solver_states.py

import argparse
import torch
import traceback


def remove_solver_state(model_in: str, model_out: str):
    print("Loading model from: %s" % model_in)
    model = torch.load(model_in)
    del model["trainer"]
    del model["iteration"]
    print("Saving modified model to: %s" % model_out)
    torch.save(model, model_out)


def main(args=None):
    """
    Performs the model building/evaluation.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Detectron2 - Remove solver state from model checkpoint to reduce file size',
                                     prog="d2_remove_solver_state",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_in', metavar='FILE', required=True, help='The model file to load process.')
    parser.add_argument('--model_out', metavar='FILE', required=True, help='The file to save the slimmed down model to.')
    parsed = parser.parse_args(args=args)

    remove_solver_state(parsed.model_in, parsed.model_out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
