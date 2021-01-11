import argparse
import json
import traceback

from pic.utils import load_state


def generate_info(state, output_format="text"):
    """
    Generates a string containing information from the model state.

    :param state: the model state to use
    :type state: dict
    :param output_format: the format to use
    :type output_format: str
    :return: the generated information
    :rtype: str
    """

    if output_format == "text":
        result = "Architecture: %s\n" % state['arch'] \
                 + "Width: %d\n" % state['width'] \
                 + "Height: %d\n" % state['height'] \
                 + "Epoch: %d\n" % state['epoch'] \
                 + "Best acc1: %.2f%%" % float(state['best_acc1']) \
                 + "Classes: %s\n" % ", ".join(state['classes']) \
                 + "# classes: %d\n" % len(state['classes'])
        if 'num_network_classes' in state:
            result += "# network classes: %d\n" % state['num_network_classes']
        return result
    elif output_format == "json":
        result = dict()
        result['architecture'] = state['arch']
        result['width'] = state['width']
        result['height'] = state['height']
        result['epoch'] = state['epoch']
        result['best_acc1'] = float(state['best_acc1'])
        result['classes'] = state['classes']
        result['num_classes'] = len(state['classes'])
        if 'num_network_classes' in state:
            result['num_network_classes'] = state['num_network_classes']
        return json.dumps(result, indent=2)
    else:
        raise Exception("Unknown format: %s" % output_format)


def info(model, output_format="text", output_file=None):
    """
    Outputs information from a model.

    :param model: the model file to query
    :type model: str
    :param output_format: the format to generate (text|json)
    :type output_format: str
    :param output_file: the file to write to, stdout if None
    :type output_file: str
    """

    print("Querying: %s" % model)
    state = load_state(model)
    info_str = generate_info(state, output_format=output_format)
    if output_file is None:
        print(info_str)
    else:
        with open(output_file, "w") as of:
            of.write(info_str)


def main(args=None):
    """
    Outputs information from a model.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(description='PyTorch Image Classification - Info',
                                     prog="pic-info",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='FILE', required=True,
                        help='The model state to output the information from')
    parser.add_argument('-f', '--format', default="text", choices=["text", "json"],
                        help='The format in which to output the information')
    parser.add_argument('-o', '--output', metavar='FILE',
                        help='The file to write the information to; outputs to stdout if not specified')
    parsed = parser.parse_args(args=args)

    info(parsed.model, output_format=parsed.format, output_file=parsed.output)


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
