ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir segmentation-models-pytorch==0.2.1 && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir opencv-python onnx "iopath>=0.1.7,<0.1.10" "fvcore>=0.1.5,<0.1.6" && \
    pip install --no-cache-dir albumentations==1.1.0 imgaug==0.4.0 && \
    pip install --no-cache-dir redis "opex==0.0.1" "redis-docker-harness==0.0.1"

WORKDIR /opt

RUN git clone https://github.com/qubvel/segmentation_models.pytorch.git segmentation_models && \
    cd segmentation_models && \
    git reset --hard a288d337821716ad67125127b5dd96a1cd833391

COPY bash.bashrc /etc/bash.bashrc

# example code
COPY examples/cars_segmentation.py /opt/segmentation_models_ext/cars_segmentation.py
COPY examples/cars_segmentation_multi.py /opt/segmentation_models_ext/cars_segmentation_multi.py

COPY common.py /opt/segmentation_models_ext/common.py
COPY predict.py /opt/segmentation_models_ext/predict.py
COPY predict_redis.py /opt/segmentation_models_ext/predict_redis.py
COPY test_image_redis.py /opt/segmentation_models_ext/test_image_redis.py
COPY train.py /opt/segmentation_models_ext/train.py

COPY sm_predict /usr/bin/sm_predict
COPY sm_predict_redis /usr/bin/sm_predict_redis
COPY sm_test_image_redis /usr/bin/sm_test_image_redis
COPY sm_train /usr/bin/sm_train
