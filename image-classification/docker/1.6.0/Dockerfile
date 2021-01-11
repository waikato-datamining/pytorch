# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# Note: don't run "apt-get upgrade -y" as the update will result in the CUDA/cuDNN backend not initializing properly??
RUN apt-get update && \
        rm -rf /var/lib/apt/lists/* && \
        pip install Cython && \
        pip install -e "git+https://github.com/waikato-datamining/pytorch.git#egg=wai.pytorchimageclass&subdirectory=image-classification" && \
        rm -Rf /root/.cache/pip

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
