ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y rustc cargo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        transformers==4.7.0 \
        datasets==1.8.0 \
        wandb==0.15.8 \
        deepspeed==0.4.0 \
        "pandas<1.5.0" \
        "protobuf==3.20.*"

COPY bash.bashrc /etc/bash.bashrc

ENV WANDB_DISABLED=true

