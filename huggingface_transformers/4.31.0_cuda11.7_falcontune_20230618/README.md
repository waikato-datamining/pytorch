# Huggingface transformers (falcontune)

Docker image for [Huggingface transformers](https://github.com/huggingface/transformers) 4.31.0
that contains [falcontune](https://github.com/rmihaylov/falcontune) support.

Uses PyTorch 2.0.1, CUDA 11.7, falcontune 20230618 (6bd029e5a89f58c4eea9056ee1c86127e1200876).

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```
* Create the `data` directory (to house downloaded dataset and generated model):

  ```bash
  mkdir data
  ```

* Launch docker container

  ```bash
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`:/workspace \
    -v `pwd`/cache:/.cache \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
```

### Build local image

* Build the image from Docker file (from within /path_to/pytorch-test/4.31.0_cuda11.7_falcontune_20230618)

  ```bash
  docker build -t huggingface_transformers .
  ```
  
* Run the container

  ```bash
  docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it huggingface_transformers
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Pre-built images

* Build

  ```bash
  docker build -t pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618 .
  ```
  
* Tag

  ```bash
  docker tag \
    pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```bash
  docker pull public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```bash
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618 \
    pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7_falcontune_20230618
  ```