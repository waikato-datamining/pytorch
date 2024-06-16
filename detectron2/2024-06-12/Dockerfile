ARG PYTORCH="2.3.0"
ARG CUDA="12.1"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2204
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
        "setuptools<=57.5.0" \
        "wheel" \
        "protobuf<4.0.0" \
        "pillow<9" \
        opencv-python \
        onnx \
        "iopath>=0.1.7,<0.1.10" \
        "fvcore>=0.1.5,<0.1.6" \
        "numpy" \
        "matplotlib"

WORKDIR /opt

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard e8806d607403cf0f2634d4c5ac464109fdc7d4af && \
    pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    pip install --no-cache-dir .

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple-file-poller>=0.0.9" \
        redis \
        "fast-opex==0.0.2" \
        orjson \
        "redis-docker-harness==0.0.4" \
        planar==0.4 \
        shapely==2.0.4

COPY bash.bashrc /etc/bash.bashrc
COPY d2_predict_common.py /opt/detectron2_ext/
COPY d2_train_coco.py /opt/detectron2_ext/
COPY d2_train_coco /usr/bin/
COPY d2_predict.py /opt/detectron2_ext/
COPY d2_predict /usr/bin/
COPY d2_predict_redis.py /opt/detectron2_ext/
COPY d2_predict_redis /usr/bin/
COPY d2_export.py /opt/detectron2_ext/
COPY d2_export /usr/bin/
COPY d2_test_image_redis.py /opt/detectron2_ext/
COPY d2_test_image_redis /usr/bin/
COPY d2_dump_config.py /opt/detectron2_ext/
COPY d2_dump_config /usr/bin/

ENV PYTHONPATH=/opt/detectron2_ext/
