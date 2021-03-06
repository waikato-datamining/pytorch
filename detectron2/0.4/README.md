# Detectron2

Uses [Detectron2](https://github.com/facebookresearch/detectron2) ([documentation](https://detectron2.readthedocs.io/en/v0.4/)). 

Uses PyTorch 1.8.1, CUDA 11.1 and Detectron2 0.4.

Though Detectron2 is installed via a wheel file, you can find Detectron2's source code \
inside the container in:

```commandline
/opt/detectron2
```

Additional code is located in:

```commandline
/opt/detectron2_ext
```

## Version

Detectron2 github repo hash:

```
4aca4bdaa9ad48b8e91d7520e0d0815bb8ca0fb1
```

and timestamp:

```
March 13, 2021
```

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run --runtime=nvidia --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.4
  ```

  **NB:** For docker versions 19.03 (`docker version`) and newer, use `--gpus=all` instead of `--runtime=nvidia`.

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```

### Build local image

* Build the image from Docker file (from within /path_to/detectron2/0.4)

  ```commandline
  sudo docker build -t detectron2 .
  ```
  
* Run the container

  ```commandline
  sudo docker run --runtime=nvidia --shm-size 8G -v /local/dir:/container/dir -it detectron2
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Pre-built images

* Build

  ```commandline
  docker build -t pytorch/detectron2:0.4 .
  ```
  
* Tag

  ```commandline
  docker tag \
    detectron2:0.4 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.4
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.4
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.4
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.4 \
    pytorch/detectron2:0.4
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia --shm-size 8G \
    -v /local/dir:/container/dir -it pytorch/detectron2:0.4
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```commandline
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```

## Caching

Detectron2 will download pretrained models and cache them locally. To avoid having
to download them constantly, you can the cache directory to the host machine:

* when running the container as `root`

  ```commandline
  -v /some/where/cache:/root/.torch \
  ```

* when running the container as current user

  ```commandline
  -v /some/where/cache:/.torch \
  ```


## Scripts

The following additional scripts are available:

* `d2_train_coco` - for building a model using a COCO-based train/test set (calls `/opt/detectron2_ext/d2_train_coco.py`)
* `d2_predict` - for generating batch predictions on images (calls `/opt/detectron2_ext/d2_predict.py`)

### d2_train_coco

* Documentation of config file parameters:

  https://detectron2.readthedocs.io/en/latest/modules/config.html
  
* Use the following in the YAML config file for the datasets (the script registers the datasets you provide via parameters under these names):

  ```yaml
  DATASETS:
    TRAIN: ("coco_ext_train",)
    TEST: ("coco_ext_test",)
  ```

* `Loss became infinite or NaN at iteration=X`
  
  Decreasing the learning rate may help (see discussion [here](https://github.com/facebookresearch/detectron2/issues/550#issuecomment-655127445)).
