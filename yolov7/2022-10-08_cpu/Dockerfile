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

RUN pip install --no-cache-dir onnx==1.10.2 onnxruntime==1.10.0 && \
    pip install --no-cache-dir "opex==0.0.1" && \
    pip install --no-cache-dir python-image-complete && \
    pip install --no-cache-dir "wai.annotations<=0.3.5" && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.1" && \
    pip install --no-cache-dir torch==1.10.1 torchvision==0.11.2 -f https://torch.kmtea.eu/whl/stable.html && \
    pip install --no-cache-dir -U numpy==1.22.1 && \
    # adapted from yolov7 requirements.txt
    pip install --no-cache-dir matplotlib>=3.2.2 \
                               opencv-python>=4.1.1 \
                               Pillow>=7.1.2 \
                               PyYAML>=5.3.1 \
                               requests>=2.23.0 \
                               scipy>=1.4.1 \
                               torch==1.10.1 \
                               torchvision==0.11.2 \
                               tqdm>=4.41.0 \
                               protobuf<4.21.3 \
                               tensorboard>=2.4.1 \
                               pandas>=1.1.4 \
                               seaborn>=0.11.0 \
                               psutil \
                               thop

WORKDIR /opt
RUN git clone https://github.com/WongKinYiu/yolov7.git && \
    cd yolov7 && \
    git reset --hard 072f76c72c641c7a1ee482e39f604f6f8ef7ee92

COPY bash.bashrc /etc/bash.bashrc
COPY yolov7_* /usr/local/bin/

ENV PYTHONPATH=/opt/yolov7
