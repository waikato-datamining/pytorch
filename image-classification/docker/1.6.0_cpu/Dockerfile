# Ubuntu 18.04, Python 3.7
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:bionic

RUN apt-get update && \
        apt-get upgrade -y && \
        apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          python3.7 \
          python3-setuptools \
          python3-pip && \
        pip3 --no-cache-dir install --upgrade pip setuptools && \
        rm -rf /var/lib/apt/lists/* && \
        ln -s /usr/bin/pip3 /usr/bin/pip && \
        ln -s /usr/bin/python3 /usr/bin/python && \
        pip install Cython && \
        pip install wai.pytorchimageclass==0.0.1 && \
        rm -Rf /root/.cache/pip && \
        rm -rf /var/lib/apt/lists/*

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
