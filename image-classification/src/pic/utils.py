import os
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def save_checkpoint(state, is_best, filename=None, output_dir="."):
    if filename is None:
        filename = 'checkpoint-%s.pth' % str(state['epoch'])
    checkpoint_filename = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_dir, 'model_best.pth')
        shutil.copyfile(checkpoint_filename, best_filename)


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

    model = models.__dict__[state['arch']](num_classes=state['num_network_classes'])
    # remove "module." from keys introduced by nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
    for k in list(state['state_dict'].keys()):
        if "module." in k:
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


def state_to_dims(state):
    """
    Turns the state into image dimensions (widht, height).

    :param state: the state to use
    :type state: dict
    :return: the width/height, tuple
    :rtype: transforms
    """
    return state['width'], state['height']
