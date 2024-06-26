# Segment-Anything in High Quality (SAM-HQ)

Command-line utilities for using [SAM-HQ](https://github.com/SysCV/sam-hq) models. 

Uses Segment-Anything 2023-08-17 (1db02cad10e4bee154b32fdc1565850332b322f6), CPU and torch 1.9.1.


## Quick start

### Inhouse registry

* Log into registry with the appropriate credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam-hq:2023-08-17-1_cpu
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/pytorch-sam-hq:2023-08-17-1_cpu
  ```

### Build local image

* Build the image from Docker file (from within `/path_to/2023-08-17-1_cpu`)

  ```bash
  docker build -t sam-hq .
  ```
  
* Run the container

  ```bash
  docker run \
    --shm-size 8G \
    -v /local/dir:/container/dir -it sam-hq
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Publish images

### Build

```bash
docker build -t pytorch-sam-hq:2023-08-17-1_cpu .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    pytorch-sam-hq:2023-08-17-1_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam-hq:2023-08-17-1_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam-hq:2023-08-17-1_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    pytorch-sam-hq:2023-08-17-1_cpu \
    waikatodatamining/pytorch-sam-hq:2023-08-17-1_cpu
  ```
  
* Push

  ```bash
  docker push waikatodatamining/pytorch-sam-hq:2023-08-17-1_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ``` 


### Requirements

```bash
docker run --rm \
  -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-sam-hq:2023-08-17-1_cpu \
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

* `samhq_predict_poll` - batch-processing of images via file-polling
* `samhq_predict_redis` - making predictions via Redis backend
* `samhq_test_redis_send` - for sending an image and extreme points to the `samhq_predict_redis` process 
* `samhq_test_redis_recv` - for receiving the results from the `samhq_predict_redis` process (and saving them to a dir) 


### samhq_predict_redis
 
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


## Prompt files

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

Pretrained models are available from Hugging Face:

https://huggingface.co/lkeab/hq-sam/tree/main

Specifically:
* [vit_h](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_h.pth)
* [vit_l](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth)
* [vit_b](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_b.pth)
* [vit_tiny](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_tiny.pth)
