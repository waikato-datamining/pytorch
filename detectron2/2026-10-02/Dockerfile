ARG PYTORCH="2.10.0"
ARG CUDA="12.6"
ARG CUDNN="9"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1 libglx-mesa0 python3-dev python3-pip python3-venv git wget build-essential cmake ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/detectron2.venv

RUN /opt/detectron2.venv/bin/pip install -U --no-cache-dir \
        "setuptools" \
        "wheel" \
        "Pillow>=7.1" \
        matplotlib \
        "pycocotools>=2.0.2" \
        opencv-python \
        onnx \
        "termcolor>=1.1" \
        "yacs>=0.1.8" \
        "tabulate" \
        "cloudpickle" \
        "tqdm>4.29.0" \
        "tensorboard" \
        "fvcore>=0.1.5,<0.1.6" \
        "iopath>=0.1.7,<0.1.10" \
        "dataclasses; python_version<'3.7'" \
        "omegaconf>=2.1,<2.4" \
        "hydra-core>=1.1" \
        "black" \
        "packaging" \
        "fairscale" \
        "timm" \
        "scipy>1.5.1" \
        "shapely" \
        "pygments>=2.2" \
        "psutil" \
        "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip"

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard 1e3e13bbf607b54f62205c4c33922521822fb298 && \
    /opt/detectron2.venv/bin/pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    /opt/detectron2.venv/bin/pip install --no-cache-dir .

RUN /opt/detectron2.venv/bin/pip install --no-cache-dir \
        python-image-complete \
        "simple-file-poller>=0.0.11" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

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
COPY d2_remove_solver_state.py /opt/detectron2_ext/
COPY d2_remove_solver_state /usr/bin/

ENV PYTHONPATH=/opt/detectron2_ext/
