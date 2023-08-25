ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
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

RUN pip install --no-cache-dir transformers==4.31.0
RUN pip install --no-cache-dir timm==0.9.2 xformers==0.0.20 && \
    pip install --no-cache-dir einops==0.6.1 accelerate==0.20.3 && \
    # https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one
    pip install --no-cache-dir bitsandbytes>=0.39.0 && \
    pip install --no-cache-dir git+https://github.com/huggingface/accelerate.git@a2d8f540c3ab37c8f84d616be1300a0572b69cf8 && \
    # https://huggingface.co/blog/falcon#fine-tuning-with-peft
    pip install --no-cache-dir scipy==1.11.1 && \
    pip install --no-cache-dir datasets==2.13.1 && \
    pip install --no-cache-dir peft==0.4.0 && \
    pip install --no-cache-dir trl==0.4.7 && \
    pip install --no-cache-dir deepspeed==0.10.1

COPY bash.bashrc /etc/bash.bashrc
