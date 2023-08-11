# pytorch-test

Simple test container that builds an image classification model
using the CIFAR10 challenge data. Code based on pytorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Can be used to test a GPU machine from within a docker container.

Uses PyTorch 2.0.1, CUDA 11.7.

## Quick start

### Inhouse registry

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```
  
* Create the `data` directory (to house downloaded dataset and generated model):

  ```bash
  mkdir data
  ```

* Launch docker container and execute the script

  ```bash
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`/data:/opt/pytorchtest/data \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch2.0.1-cuda11.7 \
    /usr/bin/pytorchtest
  ```
  
  **Notes:**

    * The first output should be `cuda:0` when the script runs on the GPU
    * 2 epochs of 12000 iterations should run

### Docker hub
  
* Create the `data` directory (to house downloaded dataset and generated model):

  ```bash
  mkdir data
  ```

* Launch docker container and execute the script

  ```bash
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`/data:/opt/pytorchtest/data \
    -it waikatodatamining/pytorchtest:pytorch2.0.1-cuda11.7 \
    /usr/bin/pytorchtest
  ```
  
  **Notes:**

    * The first output should be `cuda:0` when the script runs on the GPU
    * 2 epochs of 12000 iterations should run

### Build local image

* Build the image from Docker file (from within /path_to/pytorch-test/pytorch2.0.1-cuda11.7)

  ```bash
  docker build -t pytorchtest .
  ```
  
* Run the container

  ```bash
  docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it pytorchtest
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Publish images

### Build

```bash
docker build -t pytorchtest:pytorch2.0.1-cuda11.7 .
```

### Inhouse registry  
  
* Tag

  ```bash
  docker tag \
    pytorchtest:pytorch2.0.1-cuda11.7 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch2.0.1-cuda11.7
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorchtest:pytorch2.0.1-cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  
  
* Tag

  ```bash
  docker tag \
    pytorchtest:pytorch2.0.1-cuda11.7 \
    waikatodatamining/pytorchtest:pytorch2.0.1-cuda11.7
  ```
  
* Push

  ```bash
  docker push waikatodatamining/pytorchtest:pytorch2.0.1-cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ```
