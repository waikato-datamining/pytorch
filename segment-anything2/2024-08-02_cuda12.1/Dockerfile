ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "wheel" \
        "tqdm==4.66.1" \
        "hydra-core==1.3.2" \
        "iopath==0.1.10" \
        "pillow==9.4.0" \
        "ninja==1.11.1."

RUN pip3 install --no-cache-dir \
        "torch==2.3.1+cu121" \
        "torchvision==0.18.1+cu121" \
        "torchaudio==2.3.1+cu121" \
        --index-url https://download.pytorch.org/whl/cu121

WORKDIR /opt

# https://github.com/facebookresearch/segment-anything-2/issues/72#issue-2440198237
RUN pip3 install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder SAM_2==1.0+86827e2pt2.4.0cu121

RUN pip3 install --no-cache-dir \
        python-image-complete \
        "simple-file-poller>=0.0.9" \
        redis \
        "fast-opex==0.0.2" \
        orjson \
        "redis-docker-harness==0.0.4" \
        planar==0.4 \
        shapely==2.0.4

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/segment-anything2/
COPY test_redis*.py /opt/segment-anything2/
COPY sam2_* /usr/local/bin/

ENV PYTHONPATH=/opt/segment-anything2
