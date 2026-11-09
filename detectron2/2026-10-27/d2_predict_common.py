from typing import List, Dict
from shapely import simplify, Polygon


def load_labels(labels_file):
    """
    Loads the labels from the specified file.

    :param labels_file: the file to load (comma-separated list)
    :type labels_file: str
    :return: the list of labels
    :rtype: list
    """
    with open(labels_file) as lf:
        line = lf.readline()
    return line.strip().split(",")


def labels_to_mapping(labels: List[str]) -> Dict[int, str]:
    """
    Generates a label mapping from the list of labels.

    :param labels: the labels to turn into a lookup
    :type labels: list
    :return: the mapping
    :rtype: dict
    """
    result = dict()
    for i, label in enumerate(labels):
        result[i] = label
    return result
