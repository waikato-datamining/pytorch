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
          python3-venv \
          pkg-config \
          gnupg \
          curl && \
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
ENV NUMBA_CACHE_DIR=/tmp

ENTRYPOINT ["/bin/bash"]
