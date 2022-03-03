# pytorch-test

Simple test container that builds an image classification model
using the CIFAR10 challenge data. Code based on pytorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Can be used to test a GPU machine from within a docker container.

Uses PyTorch 1.8.1, CUDA 11.1.

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```
* Create the `data` directory (to house downloaded dataset and generated model):

  ```commandline
  mkdir data
  ```

* Launch docker container and execute the script

  ```commandline
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`/data:/opt/pytorchtest/data \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch1.8.1-cuda11.1 \
    /usr/bin/pytorchtest
  ```
  
  **Notes:**

    * The first output should be `cuda:0` when the script runs on the GPU
    * 2 epochs of 12000 iterations should run

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/pytorchtest:pytorch1.8.1-cuda11.1
```

### Build local image

* Build the image from Docker file (from within /path_to/pytorch-test/pytorch1.8.1-cuda11.1)

  ```commandline
  sudo docker build -t pytorchtest .
  ```
  
* Run the container

  ```commandline
  sudo docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it pytorchtest
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Pre-built images

* Build

  ```commandline
  docker build -t pytorch/pytorchtest:pytorch1.8.1-cuda11.1 .
  ```
  
* Tag

  ```commandline
  docker tag \
    pytorch/pytorchtest:pytorch1.8.1-cuda11.1 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch1.8.1-cuda11.1
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch1.8.1-cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch1.8.1-cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch1.8.1-cuda11.1 \
    pytorch/pytorchtest:pytorch1.8.1-cuda11.1
  ```
