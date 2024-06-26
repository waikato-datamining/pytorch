ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}debian:bullseye

ENV DEBIAN_FRONTEND noninteractive

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
          gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      onnx==1.10.2 onnxruntime==1.10.0 \
      "pillow<9" "numpy<2.0.0" "opencv-python>=4.1.1" \
      "fast-opex==0.0.2" orjson \
      python-image-complete \
      "wai.annotations<=0.3.5" \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4" && \
    pip install --no-cache-dir torch==1.9.1 torchvision==0.10.1 -f https://torch.kmtea.eu/whl/stable.html

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
