ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}debian:bullseye

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          python3-setuptools \
          git \
          python3-pip \
          python3-opencv \
          pkg-config \
          gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==1.9.1 torchvision==0.10.1 -f https://torch.kmtea.eu/whl/stable.html && \
    pip install --no-cache-dir super-gradients==3.1.1 && \
    pip install --no-cache-dir "opex==0.0.2" && \
    pip install --no-cache-dir python-image-complete && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.2"

WORKDIR /opt

COPY bash.bashrc /etc/bash.bashrc
#COPY predict*.py /opt/super-gradients/
#COPY test_redis*.py /opt/super-gradients/
#COPY sg_* /usr/local/bin/

ENV PYTHONPATH=/opt/super-gradients
