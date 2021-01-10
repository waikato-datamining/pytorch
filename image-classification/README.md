# Image classification

The **pytorch-image-classification** command-line tools can be used for building 
image classification models with PyTorch using various model architectures.

The code is base on the PyTorch imagenet example code:

https://github.com/pytorch/examples/tree/master/imagenet

Specifically, commit **49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de**:

https://github.com/pytorch/examples/tree/49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de/imagenet

## Tools

### Train

For training models, either from scratch or using transfer learning, you can use the
`pic-main` command-line utility. The tool expects the train and test directory to 
contain sub-directories which get interpreted as the categories of the images they
contain.

```commandline
usage: pic-main [-h] -t DIR -T DIR [-o DIR] [-i INT] [--width WIDTH]
                [--height HEIGHT] [-a ARCH] [-j N] [--epochs N]
                [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W]
                [-p N] [--resume PATH] [-e] [--pretrained]
                [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                [--multiprocessing-distributed]

PyTorch Image Classification - Training

optional arguments:
  -h, --help            show this help message and exit
  -t DIR, --train_dir DIR
                        path to top-level directory of training set, with each
                        sub-directory being treated as a category (default:
                        None)
  -T DIR, --test_dir DIR
                        path to top-level directory of test, with each sub-
                        directory being treated as a category (default: None)
  -o DIR, --output_dir DIR
                        the directory to store the models and checkpoints in
                        (default: .)
  -i INT, --output_interval INT
                        the output interval in epochs for checkpoints. Use -1
                        to always overwrite last checkpoint. (default: -1)
  --width WIDTH         The image width to scale to (default: 256)
  --height HEIGHT       The image height to scale to (default: 256)
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 90)
  --start-epoch N       manual epoch number (useful on restarts) (default: 0)
  -b N, --batch-size N  mini-batch size, this is the total batch size of all
                        GPUs on the current node when using Data Parallel or
                        Distributed Data Parallel (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 1e-4) (default: 0.0001)
  -p N, --print-freq N  print frequency (default: 10) (default: 10)
  --resume PATH         path to latest checkpoint (default: none) (default: )
  -e, --evaluate        evaluate model on validation set (default: False)
  --pretrained          use pre-trained model (default: False)
  --world-size WORLD_SIZE
                        number of nodes for distributed training (default: -1)
  --rank RANK           node rank for distributed training (default: -1)
  --dist-url DIST_URL   url used to set up distributed training (default:
                        tcp://224.66.41.62:23456)
  --dist-backend DIST_BACKEND
                        distributed backend (default: nccl)
  --seed SEED           seed for initializing training. (default: None)
  --gpu GPU             GPU id to use. (default: None)
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training (default: False)
```

### Predict (single image)

For applying a built model to a single image, use the `pic-predict` command-line utility:

```commandline
usage: pic-predict [-h] -m FILE -i FILE [--top_x INT]

PyTorch Image Classification - Prediction

optional arguments:
  -h, --help            show this help message and exit
  -m FILE, --model FILE
                        The model state to use (default: None)
  -i FILE, --image FILE
                        The image to apply the model to (default: None)
  --top_x INT           The top X categories to return (default: 5)
```