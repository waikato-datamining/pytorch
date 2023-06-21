# Huggingface transformers

Docker image for [Huggingface transformers](https://github.com/huggingface/transformers) 4.30.2.

Uses PyTorch 2.0.1, CUDA 11.7.

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

* Launch docker container

  ```commandline
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`:/workspace \
    -v `pwd`/cache:/.cache \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/huggingface_transformers:4.30.2_cuda11.7
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/huggingface_transformers:4.30.2_cuda11.7
```

### Build local image

* Build the image from Docker file (from within /path_to/pytorch-test/4.30.2_cuda11.7)

  ```commandline
  docker build -t huggingface_transformers .
  ```
  
* Run the container

  ```commandline
  docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it huggingface_transformers
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Pre-built images

* Build

  ```commandline
  docker build -t pytorch/huggingface_transformers:4.30.2_cuda11.7 .
  ```
  
* Tag

  ```commandline
  docker tag \
    pytorch/huggingface_transformers:4.30.2_cuda11.7 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/huggingface_transformers:4.30.2_cuda11.7
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/huggingface_transformers:4.30.2_cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/pytorch/huggingface_transformers:4.30.2_cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/pytorch/huggingface_transformers:4.30.2_cuda11.7 \
    pytorch/huggingface_transformers:4.30.2_cuda11.7
  ```
