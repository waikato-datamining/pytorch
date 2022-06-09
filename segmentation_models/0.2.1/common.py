import albumentations as albu
import importlib
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
    if ("augmentation" in config) and (key in config["augmentation"]):
        return albu.from_dict(config["augmentation"][key])
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


def instantiate_class(class_name):
    """
    Creates the specified class.

    :param class_name: the full name of the class (module and class) in dot notation
    :type class_name: str
    :return: the class
    """
    if class_name.startswith("smp."):
        class_name = "segmentation_models_pytorch." + class_name[4:]
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def instantiate_object(class_name, positional=None, parameters=None):
    """
    Instantiates an instance of the specified class, using the provided parameters for the constructor.
    
    :param class_name: the full name of the class (module and class) in dot notation
    :type class_name: str
    :param positional: list of positional arguments
    :type positional: list
    :param parameters: the named parameters for the constructor
    :type parameters: dict
    :return: the instantiated object 
    """
    cls = instantiate_class(class_name)
    has_pos = (positional is not None) and (len(positional) > 0)
    has_key = (parameters is not None) and (len(parameters) > 0)
    if has_pos and has_key:
        return cls(*positional, **parameters)
    elif has_pos:
        return cls(*positional)
    elif has_key:
        return cls(**parameters)
    else:
        return cls()


def object_from_dict(d):
    """
    Instantiates an object from the given dictionary: class, [positional:list], [parameters:dict]

    :param d: the dictionary to use
    :type d: dict
    :return: the instantiated object
    """
    class_name = d['class']
    positional = d.get('positional', None)
    parameters = d.get('parameters', None)
    return instantiate_object(class_name, positional=positional, parameters=parameters)
