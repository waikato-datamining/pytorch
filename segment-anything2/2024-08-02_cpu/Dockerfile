ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          libgl1 \
          python3-setuptools \
          git \
          python3-pip \
          pkg-config \
          gnupg \
          curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
         wheel \
        "torch==2.3.1" \
        "torchvision==0.18.1" \
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
      "fast-opex==0.0.2" orjson \
      python-image-complete \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/segment-anything2/
COPY test_redis*.py /opt/segment-anything2/
COPY sam2_* /usr/local/bin/

ENV PYTHONPATH=/opt/segment-anything2
