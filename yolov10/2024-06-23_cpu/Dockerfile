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
      torch==2.0.1 \
      torchvision==0.15.2 \
      onnx==1.14.0 \
      onnxruntime==1.15.1 \
      pycocotools==2.0.7 \
      PyYAML==6.0.1 \
      scipy==1.13.0 \
      onnxsim==0.4.36 \
      opencv-python==4.9.0.80 \
      psutil==5.9.8 \
      py-cpuinfo==9.0.0 \
      huggingface-hub==0.23.2 \
      safetensors==0.4.3 \
      "numpy<2.0.0" \
      pandas \
      seaborn
RUN pip install --no-cache-dir \
      "fast-opex==0.0.2" orjson \
      python-image-complete \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4"

WORKDIR /opt
RUN git clone https://github.com/THU-MIG/yolov10.git && \
    cd yolov10 && \
    git reset --hard aad320dd80b56694e590c950b25060a134966496 && \
    pip install --no-cache-dir --no-deps .

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/yolov10/
COPY yolov10_* /usr/local/bin/
COPY Arial.ttf /root/.config/yolov10/Arial.ttf
COPY Arial.ttf /tmp/yolov10/Arial.ttf
COPY settings.yaml /root/.config/yolov10/settings.yaml
COPY settings.yaml /tmp/yolov10/settings.yaml

ENV PYTHONPATH=/opt/yolov10
