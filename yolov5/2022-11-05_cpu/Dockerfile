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

RUN pip install --no-cache-dir onnx==1.10.2 onnxruntime==1.10.0 "pillow<9" && \
    pip install --no-cache-dir "opex==0.0.1" && \
    pip install --no-cache-dir python-image-complete && \
    pip install --no-cache-dir "wai.annotations<=0.3.5" && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4" && \
    pip install --no-cache-dir torch==1.9.1 torchvision==0.10.1 -f https://torch.kmtea.eu/whl/stable.html && \
    pip install --no-cache-dir -U numpy==1.22.1

WORKDIR /opt
RUN git clone https://github.com/ultralytics/yolov5 && \
    cd yolov5 && \
    git reset --hard fde77584687041aa62795bb2c27e895cf73686bf && \
    pip install --no-cache-dir -r requirements.txt && \
    git config --global --add safe.directory '/opt/yolov5'

COPY predict*.py /opt/yolov5/
COPY bash.bashrc /etc/bash.bashrc
COPY yolov5* /usr/local/bin/
COPY Arial.ttf /root/.config/Ultralytics/Arial.ttf
COPY Arial.ttf /tmp/Ultralytics/Arial.ttf

ENV PYTHONPATH=/opt/yolov5
