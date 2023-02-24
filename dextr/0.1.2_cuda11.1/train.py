# The MIT License (MIT)
#
# Copyright (c) 2020 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.
import click

@click.command()
@click.argument('job_name', type=str)
@click.option('--dataset', type=click.Choice(['pascal_voc', 'custom_label', 'custom_mask']),
              default='pascal_voc', help='Dataset to use')
@click.option('--train_image_pat', type=str, default='',
              help='Training images path pattern, e.g. mydataset/train/images/*.jpg')
@click.option('--train_target_pat', type=str, default='',
              help='Training targets path pattern, e.g. mydataset/train/labels/*.png')
@click.option('--val_image_pat', type=str, default='',
              help='Training images path pattern, e.g. mydataset/val/images/*.jpg')
@click.option('--val_target_pat', type=str, default='',
              help='Training targets path pattern, e.g. mydataset/val/labels/*.png')
@click.option('--label_ignore_index', type=int,
              help='Label index to ignore when using custom_label dataset (e.g. 255 is used for Pascal VOC)')
@click.option('--arch', type=click.Choice(['deeplab3', 'denseunet161', 'resunet50', 'resunet101']), default='resunet50',
              help='Architecture to use. deeplab3 uses TorchVision COCO pre-trained DeepLab3, the others '
                   'are U-nets based on ImageNet classifiers')
@click.option('--load_model', type=click.Path(readable=True),
              help='Path to load DEXTR model weights from to use as a starting point, e.g. if you want to start '
                   'with a DEXTR model pre-trained on Pascal VOC')
@click.option('--learning_rate', type=float, default=0.1,
              help='The learning rate to use. 0.1 works well for SGD')
@click.option('--pretrained_lr_factor', type=float, default=0.1,
              help='The learning rate scale factor to use for pre-trained weights. 0.1 works well.')
@click.option('--lr_sched', type=click.Choice(['none', 'cosine', 'poly']), default='poly',
              help='The type of learning rate schedule. '
                   '(polynomial works well for semantic segmentation problems, so poly is the default)')
@click.option('--lr_poly_power', type=float, default=0.9,
              help='Exponent for polynomial LR schedule')
@click.option('--opt_type', type=click.Choice(['sgd', 'adam']), default='sgd',
              help='Type of optimizer to use')
@click.option('--sgd_weight_decay', type=float, default=1e-4,
              help='Weight decay to apply when using SGD.')
@click.option('--target_size', type=int, default=512,
              help='The image size to scale the crop to for inference/training.')
@click.option('--padding', type=int, default=10,
              help='In target crop space, this amount of space will be left between the edges and the '
                   'extreme points')
@click.option('--extreme_range', type=int, default=5,
              help='During training, extreme points will be chosen from the mask within this range of pixels '
                   'of the edge of the mask bounding box.')
@click.option('--noise_std', type=float, default=1.0,
              help='During training Gaussian noise of this sigma will be added to extreme point positions')
@click.option('--blob_sigma', type=float, default=10.0,
              help='Gaussian blobs of this sigma will be placed and the positions of extreme points.')
@click.option('--aug_hflip', is_flag=True, default=False,
              help='If true, apply random horizontal flips during training.')
@click.option('--aug_rot_range', type=float, default=20.0,
              help='Rotation augmentation range in degrees; rotate samples between -aug_rot_range and aug_rot_range '
                   'degrees.')
@click.option('--batch_size', type=int, default=4,
              help='Batch size for training')
@click.option('--num_epochs', type=int, default=100,
              help='Train for this many epochs')
@click.option('--iters_per_epoch', type=int, default=1000,
              help='Number of iterations/mini-batches per epoch')
@click.option('--val_every_n_epochs', type=int, default=25,
              help='Validate every N epochs')
@click.option('--device', type=str, default='cuda:0',
              help='Torch device for training')
@click.option('--num_workers', type=int, default=8,
              help='Number of background processes for data loading')
def train_dextr(job_name, dataset, train_image_pat, train_target_pat, val_image_pat, val_target_pat, label_ignore_index,
                arch, learning_rate, load_model, pretrained_lr_factor, lr_sched, lr_poly_power,
                opt_type, sgd_weight_decay,
                target_size, padding, extreme_range, noise_std, blob_sigma, aug_hflip, aug_rot_range,
                batch_size, num_epochs, iters_per_epoch, val_every_n_epochs, device, num_workers):
    settings = locals().copy()

    import pathlib
    import tqdm
    import glob
    import torch.utils.data
    from dextr import model
    from dextr.data_pipeline import dextr_dataset, pascal_voc_dataset
    from dextr.architectures import deeplab, denseunet, resunet
    import job_output

    output = job_output.JobOutput(job_name, False)
    output.connect_streams()

    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))

    torch_device = torch.device(device)

    def match_image_paths_with_label_paths(image_paths, label_paths):
        # Convert to lists
        image_paths = list(image_paths)
        label_paths = list(label_paths)
        # Sort in reverse order of stems
        # This way if the name of one image (e.g. 'img0') is the prefix of another (e.g. 'img0b'),
        # then label files whose name start with the longer name will appear first and be matched
        # first (e.g. 'img0b_label' will be matched with 'img0b' and 'img0_label' with 'img0',
        # rather than both 'img0_label' and 'img0b_label' matching with 'img0').
        image_paths.sort(key=lambda p: p.stem, reverse=True)
        label_paths.sort(key=lambda p: p.stem, reverse=True)
        input_stems = [f.stem for f in image_paths]
        label_stems = [f.stem for f in label_paths]
        x_paths = []
        y_paths = []
        label_pos = 0
        for input_i, input_name in enumerate(input_stems):
            for label_j in range(label_pos, len(label_stems)):
                if label_stems[label_j].startswith(input_name):
                    # Match found
                    x_paths.append(str(image_paths[input_i]))
                    y_paths.append(str(label_paths[label_j]))
                    label_pos = label_j + 1
                    break
                elif label_stems[label_j] < input_name:
                    # No match available
                    label_pos = label_j
                    break
        xy_paths = list(zip(x_paths, y_paths))
        xy_paths.sort()
        x_paths = [xy[0] for xy in xy_paths]
        y_paths = [xy[1] for xy in xy_paths]
        return x_paths, y_paths

    def match_image_paths_with_mask_stack_paths(image_paths, mask_paths):
        # Convert to lists
        image_paths = list(image_paths)
        mask_paths = list(mask_paths)
        # Sort in reverse order of stems
        # This way if the name of one image (e.g. 'img0') is the prefix of another (e.g. 'img0b'),
        # then label files whose name start with the longer name will appear first and be matched
        # first (e.g. 'img0b_label' will be matched with 'img0b' and 'img0_label' with 'img0',
        # rather than both 'img0_label' and 'img0b_label' matching with 'img0').
        image_paths.sort(key=lambda p: p.stem, reverse=True)
        mask_paths.sort(key=lambda p: p.stem, reverse=True)
        input_stems = [f.stem for f in image_paths]
        mask_stems = [f.stem for f in mask_paths]
        x_paths = []
        y_paths = []
        mask_pos = 0
        for input_i, input_name in enumerate(input_stems):
            matched = False
            for mask_j in range(mask_pos, len(mask_stems)):
                if mask_stems[mask_j].startswith(input_name):
                    # Match found
                    if not matched:
                        x_paths.append(str(image_paths[input_i]))
                        y_paths.append([])
                    y_paths[-1].append(str(mask_paths[mask_j]))
                    mask_pos = mask_j + 1
                    matched = True
                elif mask_stems[mask_j] < input_name:
                    # Finished matching
                    mask_pos = mask_j
                    break
        xy_paths = list(zip(x_paths, y_paths))
        xy_paths.sort()
        x_paths = [xy[0] for xy in xy_paths]
        y_paths = [xy[1] for xy in xy_paths]
        for x, y in zip(x_paths, y_paths):
            print('{} <--> {}'.format(x, y))
        return x_paths, y_paths

    if dataset == 'pascal_voc':
        train_ds_fn = lambda transform=None: pascal_voc_dataset.PascalVOCDataset(
            'train', transform, progress_fn=tqdm.tqdm)
        val_ds_fn = lambda transform=None: pascal_voc_dataset.PascalVOCDataset(
            'val', transform, progress_fn=tqdm.tqdm)
        val_truth_ds_fn = lambda: pascal_voc_dataset.PascalVOCDataset(
            'val', None, load_input=False, progress_fn=tqdm.tqdm)
    elif dataset == 'custom_label':
        if train_image_pat == '':
            print('Training image pattern MUST be provided when using custom_label data set')
        if train_target_pat == '':
            print('Training target pattern MUST be provided when using custom_label data set')

        train_x, train_y = match_image_paths_with_label_paths(
            [pathlib.Path(p) for p in glob.glob(train_image_pat)],
            [pathlib.Path(p) for p in glob.glob(train_target_pat)])

        train_ds_fn = lambda transform=None: dextr_dataset.LabelImageTargetDextrDataset(
            train_x, train_y, transform=transform, ignore_index=label_ignore_index, progress_fn=tqdm.tqdm)

        if val_image_pat != '' and val_target_pat != '':
            val_x, val_y = match_image_paths_with_label_paths(
                [pathlib.Path(p) for p in glob.glob(val_image_pat)],
                [pathlib.Path(p) for p in glob.glob(val_target_pat)])

            val_ds_fn = lambda transform=None: dextr_dataset.LabelImageTargetDextrDataset(
                val_x, val_y, transform=transform, ignore_index=label_ignore_index, progress_fn=tqdm.tqdm)
            val_truth_ds_fn = lambda: dextr_dataset.LabelImageTargetDextrDataset(
                val_x, val_y, transform=None, ignore_index=label_ignore_index, load_input=False,
                progress_fn=tqdm.tqdm)
        else:
            val_ds_fn = None
            val_truth_ds_fn = None

    elif dataset == 'custom_mask':
        if train_image_pat == '':
            print('Training image pattern MUST be provided when using custom_mask data set')
        if train_target_pat == '':
            print('Training target pattern MUST be provided when using custom_mask data set')

        train_x, train_y = match_image_paths_with_mask_stack_paths(
            [pathlib.Path(p) for p in glob.glob(train_image_pat)],
            [pathlib.Path(p) for p in glob.glob(train_target_pat)])

        train_ds_fn = lambda transform=None: dextr_dataset.MaskStackTargetDextrDataset(
            train_x, train_y, transform=transform, progress_fn=tqdm.tqdm)

        if val_image_pat != '' and val_target_pat != '':
            val_x, val_y = match_image_paths_with_mask_stack_paths(
                [pathlib.Path(p) for p in glob.glob(val_image_pat)],
                [pathlib.Path(p) for p in glob.glob(val_target_pat)])

            val_ds_fn = lambda transform=None: dextr_dataset.MaskStackTargetDextrDataset(
                val_x, val_y, transform=transform, progress_fn=tqdm.tqdm)
            val_truth_ds_fn = lambda: dextr_dataset.MaskStackTargetDextrDataset(
                val_x, val_y, transform=None, load_input=False, progress_fn=tqdm.tqdm)
        else:
            val_ds_fn = None
            val_truth_ds_fn = None

    else:
        print('Unknown dataset {}'.format(dataset))
        return

    # Build network
    if arch == 'deeplab3':
        net = deeplab.dextr_deeplab3(1)
    elif arch == 'denseunet161':
        net = denseunet.dextr_denseunet161(1)
    elif arch == 'resunet50':
        net = resunet.dextr_resunet50(1)
    elif arch == 'resunet101':
        net = resunet.dextr_resunet101(1)
    else:
        print('Unknown network architecture {}'.format(arch))
        return

    # Load model if path provided
    if load_model is not None:
        print('Loading snapshot from {}...'.format(load_model))
        dextr_model = torch.load(load_model, map_location=torch_device)
        new_lr_factor = pretrained_lr_factor
    else:
        dextr_model = model.DextrModel(net, (target_size, target_size), padding, blob_sigma)
        new_lr_factor = 1.0

    def on_epoch_finished(epoch, model):
        output.write_checkpoint(model)

    model.training_loop(dextr_model, train_ds_fn, val_ds_fn, val_truth_ds_fn, extreme_range,
                        noise_std, learning_rate, pretrained_lr_factor, new_lr_factor,
                        lr_sched, lr_poly_power, opt_type, sgd_weight_decay, aug_hflip, aug_rot_range,
                        batch_size, iters_per_epoch, num_epochs, val_every_n_epochs,
                        torch_device, num_workers, True, on_epoch_finished)


if __name__ == '__main__':
    train_dextr()
