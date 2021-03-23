ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir opencv-python onnx

WORKDIR /opt

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard 4841e70ee48da72c32304f9ebf98138c2a70048d && \
    cd /opt/detectron2/projects/TensorMask && \
    pip install --no-cache-dir .

COPY bash.bashrc /etc/bash.bashrc
COPY d2_train_coco.py /opt/detectron2_ext/d2_train_coco.py
COPY d2_train_coco /usr/bin/d2_train_coco
COPY d2_predict.py /opt/detectron2_ext/d2_predict.py
COPY d2_predict /usr/bin/d2_predict
