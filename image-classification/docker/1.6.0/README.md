# Labeling images

Allows labeling of images with PyTorch's image classification capabilities, using PyTorch 1.6.0.

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run --gpus=all \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/image-classification:1.6
  ```

  **NB:** For docker versions older than 19.03 (`docker version`), use `--runtime=nvidia` instead of `--gpus=all`.

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```


### Build local image

* Build the image from Docker file (from within /path_to/pytorch/image-classification/docker/1.6)

  ```commandline
  docker build -t tfic .
  ```

* Run the container

  ```commandline
  docker run --gpus=all -v /local/dir:/container/dir -it tfic
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t pytorch/image-classification:1.6 .
  ```
  
* Tag

  ```commandline
  docker tag \
    pytorch/image-classification:1.6 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/image-classification:1.6
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/image-classification:1.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/pytorch/image-classification:1.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/pytorch/image-classification:1.6 \
    pytorch/image-classification:1.6
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --gpus=all -v /local/dir:/container/dir -it pytorch/image-classification:1.6
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Usage

The following command-line tools are available (see [here](../../README.md) for more details):

* `pic-main` - for training a (pre-trained) model on data
* `pic-predict` - labeling a single image
* `pic-poll` - for batch or continuous predictions
* `pic-export` - for exporting a model to [TorchScript](https://pytorch.org/docs/stable/jit.html)
* `pic-info` - outputs information about a trained model
