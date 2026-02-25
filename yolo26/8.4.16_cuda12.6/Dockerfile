FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 python3-dev python3-pip python3-venv git wget build-essential cmake ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/yolo26
RUN /opt/yolo26/bin/pip install --no-cache-dir \
     ultralytics-opencv-headless==8.4.16
RUN . /opt/yolo26/bin/activate && \
    /opt/yolo26/bin/pip install --no-cache-dir \
      "fast-opex==0.0.4" orjson \
      python-image-complete \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY yolo26_* /usr/local/bin/
COPY predict_*.py /opt/yolo26-extra/

ENV PYTHONPATH=/opt/yolo26-extra
ENV TORCHINDUCTOR_CACHE_DIR=/cache
ENV YOLO_CONFIG_DIR=/config

ENTRYPOINT ["/bin/bash"]
