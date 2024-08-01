ARG PYTORCH="2.3.1"
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
        "wheel" \
        "tqdm==4.66.1" \
        "hydra-core==1.3.2" \
        "iopath==0.1.10" \
        "pillow==9.4.0"

WORKDIR /opt

RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && \
    git reset --hard 0e78a118995e66bb27d78518c4bd9a3e95b4e266 && \
    pip -v install --no-cache-dir .

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
COPY predict*.py /opt/segment-anything2/
COPY test_redis*.py /opt/segment-anything2/
COPY sam2_* /usr/local/bin/

ENV PYTHONPATH=/opt/segment-anything2
