# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

# Note: don't run "apt-get upgrade -y" as the update will result in the CUDA/cuDNN backend not initializing properly??
RUN apt-get update && \
        apt-get -y install git python3-pip python3.7 && \
        rm -rf /var/lib/apt/lists/* && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 && \
        pip3 --no-cache-dir install --upgrade pip setuptools && \
        pip3 install Cython && \
        pip3 install "wai.pytorchimageclass==0.0.2" && \
        rm -Rf /root/.cache/pip

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
