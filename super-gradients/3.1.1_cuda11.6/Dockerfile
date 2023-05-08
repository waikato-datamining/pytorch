ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          python3-setuptools \
          git \
          python3-opencv \
          pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir super-gradients==3.1.1 && \
    pip install --no-cache-dir "opex==0.0.2" && \
    pip install --no-cache-dir python-image-complete && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.1"

COPY bash.bashrc /etc/bash.bashrc
#COPY predict*.py /opt/super-gradients/
#COPY test_redis*.py /opt/super-gradients/
#COPY sg_* /usr/local/bin/

ENV PYTHONPATH=/opt/super-gradients
