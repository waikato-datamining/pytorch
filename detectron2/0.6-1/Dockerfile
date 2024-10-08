ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir detectron2==0.6 "protobuf<4.0.0" "pillow<9" "scikit-image<0.20" "matplotlib<3.7" \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.11" && \
    pip install --no-cache-dir opencv-python onnx "iopath>=0.1.7,<0.1.10" "fvcore>=0.1.5,<0.1.6"

RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /opt

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard d1e04565d3bec8719335b88be9e9b961bf3ec464 && \
    pip install --no-cache-dir "numpy<1.20" "pillow<9.0" "matplotlib<3.7" && \
    pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    pip install --no-cache-dir .

RUN pip install --no-cache-dir redis "fast-opex==0.0.4" orjson "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY d2_train_coco.py /opt/detectron2_ext/d2_train_coco.py
COPY d2_train_coco /usr/bin/d2_train_coco
COPY d2_predict.py /opt/detectron2_ext/d2_predict.py
COPY d2_predict /usr/bin/d2_predict
COPY d2_predict_redis.py /opt/detectron2_ext/d2_predict_redis.py
COPY d2_predict_redis /usr/bin/d2_predict_redis
COPY d2_export.py /opt/detectron2_ext/d2_export.py
COPY d2_export /usr/bin/d2_export
COPY d2_test_image_redis.py /opt/detectron2_ext/d2_test_image_redis.py
COPY d2_test_image_redis /usr/bin/d2_test_image_redis
COPY d2_dump_config.py /opt/detectron2_ext/d2_dump_config.py
COPY d2_dump_config /usr/bin/d2_dump_config
