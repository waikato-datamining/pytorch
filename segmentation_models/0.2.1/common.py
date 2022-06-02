import albumentations as albu
import json
import yaml


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_augmentation(config, key):
    """
    Instantiates the albumentation augmentation from the configuration dictionary.

    :param config: the configuration dictionary to get the augmentation from
    :type config: dict
    :param key: the key in the configuration with the augmentation
    :type key: str
    :return: the albumentation augmentation
    """
    if key in config:
        return albu.from_dict(config[key])
    else:
        return albu.Compose([])


def load_config(cfg):
    """
    Loads the configuration from the specified json or yaml file.

    :param cfg: the config file to load (.json, .yaml, .yml)
    :type cfg: str
    :return: the configuration dictionary
    :rtype: dict
    """
    if cfg.lower().endswith(".json"):
        with open(cfg, "r") as fp:
            result = json.load(fp)
    elif cfg.lower().endswith(".yaml") or cfg.lower().endswith(".yml"):
        with open(cfg, "r") as fp:
            result = yaml.safe_load(fp)
    else:
        raise Exception("Unhandled file format: %s" % cfg)

    # convert epochs to int
    if 'lr_schedule' in result:
        schedule = {}
        for k in result['lr_schedule']:
            schedule[int(k)] = result['lr_schedule'][k]
        result['lr_schedule'] = schedule

    return result