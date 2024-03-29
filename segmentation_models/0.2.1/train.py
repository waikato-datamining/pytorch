import argparse
import cv2
import json
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
import traceback

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from common import get_preprocessing, get_augmentation, load_config, instantiate_object, instantiate_class, object_from_dict


class Dataset(BaseDataset):
    """
    Dataset for image segmentation.
    """

    def __init__(self, dtype, images_dir, classes, classes_to_use=None, augmentation=None, preprocessing=None, verbose=False):
        """
        Read images, apply augmentation and preprocessing transformations.

        :param dtype: the dataset type (eg train/test/val)
        :type dtype: str
        :param images_dir: the directory with the images (jpg: actual, png: mask)
        :type images_dir: str
        :param classes: the list of labels corresponding to the annotations (excluding background, ie color #000000)
        :type classes: list
        :param classes_to_use: the list of labels to actually use, uses all if None
        :type classes_to_use: list
        :param augmentation: the albumentation augmentations to apply
        :param preprocessing: the preprocessing to apply
        :param verbose: whether to output some debugging information
        :type verbose: bool
        """
        self.dtype = dtype
        self.verbose = verbose
        self.ids = []
        self.images = []
        for f in os.listdir(images_dir):
            if f.lower().endswith(".jpg"):
                mask_prefix = os.path.join(images_dir, os.path.splitext(f)[0])
                if os.path.exists(mask_prefix + ".png") or os.path.exists(mask_prefix + ".PNG"):
                    self.ids.append(f)
                    self.images.append(os.path.join(images_dir, f))
                else:
                    print("%s has no PNG mask associated (.png or .PNG)!" % f)

        # convert str names to class values on masks
        self.classes = self._lower(classes)
        self.classes_to_use = self.classes if (classes_to_use is None) else self._lower(classes_to_use)
        self.classes_to_use_indices = [self.classes.index(cls) for cls in self.classes_to_use]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        if self.verbose:
            print("%s - # ids: %d" % (self.dtype, len(self.ids)))
            print("%s - classes: %s" % (self.dtype, str(self.classes)))
            print("%s - classes to use: %s" % (self.dtype, str(self.classes_to_use)))
            print("%s - classes to use/indices: %s" % (self.dtype, str(self.classes_to_use_indices)))

    def _lower(self, labels):
        """
        Turns the labels in the list to lower case.
        
        :param labels: the list to convert to lower case
        :type labels: list
        :return: the new list
        :rtype: list
        """
        if labels is None:
            return None
        else:
            return [x.lower() for x in labels]

    def __getitem__(self, i):
        """
        Returns the referenced image/mask tuple in the dataset.

        :param i: the index in the dataset to retrieve
        :type i: int
        :return: the image/mask tuple
        :rtype: tuple
        """
        # if self.verbose:
        #     print("%s - getitem: %s" % (self.dtype, self.ids[i]))

        # read data
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_prefix = os.path.splitext(self.images[i])[0]
        if os.path.exists(mask_prefix + ".png"):
            mask = cv2.imread(mask_prefix + ".png", cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(mask_prefix + ".PNG", cv2.IMREAD_GRAYSCALE)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.classes_to_use_indices]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """
        Returns the number of images in the dataset.

        :return: the number of images
        :rtype: int
        """
        return len(self.ids)


def save_log(log, output_dir, prefix, epoch=None):
    """
    Saves the log in the specified output directory.

    :param log: the log to save
    :param output_dir: the output directory
    :type output_dir: str
    :param prefix: the prefix to use, eg 'train-'
    :type prefix: str
    :param epoch: the current epoch
    :type epoch: int
    """
    if epoch is not None:
        output_file = os.path.join(output_dir, "%s%d.json" % (prefix, epoch))
    else:
        output_file = os.path.join(output_dir, "%s.json" % prefix)
    try:
        with open(output_file, "w") as fp:
            json.dump(log, fp, indent=2)
    except:
        print("Failed to store log: %s\n%s" % (output_file, traceback.format_exc()))


def train(train_dir, val_dir, output_dir, config, test_dir=None, device='cuda', verbose=False):
    """
    Method for performing predictions on images.

    :param train_dir: the directory with .jpg images and .png masks to use for training
    :type train_dir: str
    :param val_dir: the directory with .jpg images and .png masks to use for validation
    :type val_dir: str
    :param output_dir: the directory to store the model and other files in
    :type output_dir: str
    :param config: the configuration dictionary
    :type config: dict
    :param test_dir: the directory with .jpg images and .png masks to use for testing the final model
    :type test_dir: str
    :param device: the device to perform the training on, eg 'cuda' or 'cpu'
    :type device: str
    :param verbose: whether to output more logging information
    :type verbose: bool
    """

    model_params = config['model'].get('parameters', None)
    model = object_from_dict(config['model'])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_params['encoder_name'], model_params['encoder_weights'])
    preprocessing = get_preprocessing(preprocessing_fn)

    # augmentations
    train_transform = get_augmentation(config, 'train')
    test_transform = get_augmentation(config, 'test')

    # datasets
    train = Dataset("train", train_dir, config['classes'], classes_to_use=config['classes_to_use'], augmentation=train_transform, preprocessing=preprocessing, verbose=verbose)
    val = Dataset("val", val_dir, config['classes'], classes_to_use=config['classes_to_use'], augmentation=test_transform, preprocessing=preprocessing, verbose=verbose)
    test = None
    if test_dir is not None:
        test = Dataset("val", test_dir, config['classes'], classes_to_use=config['classes_to_use'], augmentation=test_transform, preprocessing=preprocessing, verbose=verbose)

    # train
    train_config = config['train']
    train_loader = DataLoader(train, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'])
    valid_loader = DataLoader(val, batch_size=config['validate']['batch_size'], shuffle=False, num_workers=config['validate']['num_workers'])

    loss = object_from_dict(train_config['loss'])
    metrics = []
    for m in train_config['metrics']:
        if isinstance(m, dict):
            metrics.append(object_from_dict(m))
        else:
            metrics.append(m)
    optimizer_class = instantiate_class(train_config['optimizer']['class'])
    optimizer = optimizer_class([
        dict(params=model.parameters(), **train_config['optimizer']['parameters'])
    ])
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )
    # train model for 40 epochs
    max_score = 0
    lr_schedule = {} if ('lr_schedule' not in train_config) else train_config['lr_schedule']
    all_train = []
    all_valid = []
    for i in range(train_config['num_epochs']):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        all_train.append({"epoch": i, "log": train_logs})
        save_log(all_train, output_dir, "train")

        valid_logs = valid_epoch.run(valid_loader)
        all_valid.append({"epoch": i, "log": valid_logs})
        save_log(all_valid, output_dir, "val")

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(output_dir, 'best_model.pth'))
            print('Model saved!')

        if i in lr_schedule:
            optimizer.param_groups[0]['lr'] = lr_schedule[i]
            print('Decreasing decoder learning rate to: %f' % lr_schedule[i])

    # test best model
    if test is not None:
        # load best saved checkpoint
        best_model = torch.load(os.path.join(output_dir, 'best_model.pth'))
        test_dataloader = DataLoader(test)
        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=device,
        )
        logs = test_epoch.run(test_dataloader)
        save_log(logs, output_dir, "test")


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Segmentation Models - Prediction',
                                     prog="sm_predict",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', metavar='DIR', required=True, help='The directory with .jpg images and .png masks to use for training')
    parser.add_argument('--val_dir', metavar='DIR', required=True, help='The directory with .jpg images and .png masks to use for validation')
    parser.add_argument('--test_dir', metavar='DIR', default=None, help='The directory with .jpg images and .png masks to use for testing the final model')
    parser.add_argument('--output_dir', metavar='DIR', required=True, help='The directory to store the model and other files in')
    parser.add_argument('--config', metavar='FILE', required=True, help='The configuration in JSON (.json) or YAML (.yaml, .yml) format')
    parser.add_argument('--device', metavar='DEVICE', default="cuda", help='The device to use for inference, like "cpu" or "cuda"')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load config
    print("Loading config...")
    config = load_config(parsed.config)

    train(parsed.train_dir, parsed.val_dir, parsed.output_dir, config,
          test_dir=parsed.test_dir, device=parsed.device, verbose=parsed.verbose)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
