import torch
import torchvision.models as models
import torchvision.transforms as transforms

from pic.main import NORMALIZE


def load_state(model_filename):
    """
    Loads the specified model form disk.

    :param model_filename: the filename of the model state to load
    :type model_filename: str
    :return: the model
    :rtype: dict
    """
    print("Loading state...")
    if torch.cuda.is_available():
        return torch.load(model_filename)
    else:
        return torch.load(model_filename, map_location='cpu')


def state_to_model(state):
    """
    Restores the model from the state.

    :param state: the state to restore the model from
    :type state: dict
    :return: the model
    :rtype: object
    """

    classes = state['classes']
    model = models.__dict__[state['arch']](num_classes=len(classes))
    # remove "module." prefix in keys introduced by nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
    for k in list(state['state_dict'].keys()):
        if k.startswith("module."):
            state['state_dict'][k.replace("module.", "")] = state['state_dict'][k]
            del state['state_dict'][k]
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


def state_to_transforms(state):
    """
    Turns the state into images transformations.

    :param state: the state to use
    :type state: dict
    :return: the image transformations
    :rtype: transforms
    """
    return transforms.Compose([
        transforms.Resize((state['width'], state['height'])),
        transforms.ToTensor(),
        NORMALIZE,
    ])
