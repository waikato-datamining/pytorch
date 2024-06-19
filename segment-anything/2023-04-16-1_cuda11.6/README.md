# Segment-Anything Model

Command-line utilities for using [SAM](https://github.com/facebookresearch/segment-anything) models. 

Uses Segment-Anything 2023-04-16 (567662b0fd33ca4b022d94d3b8de896628cd32dd), CUDA 11.6 and torch 1.13.0.


## Quick start

### Inhouse registry

* Log into registry with the appropriate credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --gpus=all --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam:2023-04-16-1_cuda11.6
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --gpus=all --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/pytorch-sam:2023-04-16-1_cuda11.6
  ```

### Build local image

* Build the image from Docker file (from within `/path_to/2023-04-16-1_cuda11.6`)

  ```bash
  docker build -t sam .
  ```
  
* Run the container

  ```bash
  docker run \
    --gpus=all --shm-size 8G \
    -v /local/dir:/container/dir -it sam
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Publish images

### Build

```bash
docker build -t pytorch-sam:2023-04-16-1_cuda11.6 .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    pytorch-sam:2023-04-16-1_cuda11.6 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam:2023-04-16-1_cuda11.6
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam:2023-04-16-1_cuda11.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    pytorch-sam:2023-04-16-1_cuda11.6 \
    waikatodatamining/pytorch-sam:2023-04-16-1_cuda11.6
  ```
  
* Push

  ```bash
  docker push waikatodatamining/pytorch-sam:2023-04-16-1_cuda11.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ``` 


### Requirements

```bash
docker run --rm \
  -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam:2023-04-16-1_cuda11.6 \
  pip freeze > requirements.txt
```


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```


## Scripts

The following additional scripts are available:

* `sam_predict_poll` - batch-processing of images via file-polling
* `sam_predict_redis` - making predictions via Redis backend
* `sam_test_redis_send` - for sending an image and extreme points to the `sam_predict_redis` process 
* `sam_test_redis_recv` - for receiving the results from the `sam_predict_redis` process (and saving them to a dir) 


### sam_predict_redis
 
You need to start the docker container with the `--net=host` option if you are using the host's Redis server.

Data formats:

* sending: 

  ```
  {
    "image": base64-encoded JPG bytes,
    "prompt": prompt data structure (see below)
  }
  ```

* receiving:

  ```
  {
    "mask": base64-encoded PNG bytes,
    "contours": contours in OPEX format (https://github.com/WaikatoLink2020/objdet-predictions-exchange-format)
    "meta": {
      "key": "value"
    }
  }
  ```

**Notes:**

* `meta`: contains meta-data, like the segmenter and prompt that was used


## Prompt data structures

The JSON files for SAM prompts can have the following format:

* points (label=0: background, label=1: foreground)

  ```json
  {
    "points": [
      {
        "x": 10,
        "y": 100,
        "label": 0
      }    
    ]  
  }
  ```
  
* box

  ```json
  {
    "box": {
      "x0": 10,
      "y0": 100,
      "x1": 200,
      "y1": 150
    }  
  }
  ```
  

## Pretrained models

The following pretrained models are available:

Specifically:
* [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)