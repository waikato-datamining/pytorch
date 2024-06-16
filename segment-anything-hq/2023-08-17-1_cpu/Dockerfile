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

RUN pip install --no-cache-dir "pillow<9" torch==1.9.1 torchvision==0.10.1 -f https://torch.kmtea.eu/whl/stable.html && \
    pip install --no-cache-dir -U numpy==1.22.1 && \
    pip install --no-cache-dir onnx==1.10.2 onnxruntime==1.10.0 &&  \
    pip install --no-cache-dir "fast-opex==0.0.2" orjson && \
    pip install --no-cache-dir python-image-complete && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4" && \
    pip install --no-cache-dir "matplotlib>=3.2.2" \
                               "opencv-python>=4.1.1" \
                               "wai.pycocotools" \
                               "scikit-image==0.19.3" \
                               "timm==0.9.2"

WORKDIR /opt
RUN git clone https://github.com/SysCV/sam-hq.git && \
    cd sam-hq && \
    git reset --hard 1db02cad10e4bee154b32fdc1565850332b322f6

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/sam-hq/
COPY test_redis*.py /opt/sam-hq/
COPY samhq_* /usr/local/bin/

ENV PYTHONPATH=/opt/sam-hq
