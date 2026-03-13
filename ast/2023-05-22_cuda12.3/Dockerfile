FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git-core \
       wget \
       python3 \
       python3-dev \
       python3-venv \
       python3-pip \
       python3-numpy \
       python3-full \
       python3-soundfile \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxrender-dev \
       libxext6 \
       libgl1 \
       libglx-mesa0 \
       libpq-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/local/cuda-12.3/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.3/lib64
ENV CUDA_HOME=/usr/local/cuda-12.3

WORKDIR /opt

RUN git clone https://github.com/YuanGongND/ast.git && \
    cd ast && \
    git reset --hard 31088be8a3f6ef96416145c4b8d43c81f99eba7a
RUN python3 -m venv --system-site-packages /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir \
      "llvmlite==0.38.0" \
      "matplotlib==3.4.2" \
      "numba==0.55.1" \
      "numpy<2.0.0" \
      "scikit-learn==1.2.2" \
      "scipy==1.7.3" \
      "sklearn==0.0" \
      "timm==0.4.5" \
      "torch==1.11.0" \
      "torchaudio==0.11.0" \
      "torchvision==0.12.0" \
      "wget==3.2" \
      "zipp==3.4.1"
RUN /opt/venv/bin/pip install --no-cache-dir \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY predict_*.py /opt/ast-extra/
COPY ast_* /usr/local/bin/

ENV PYTHONPATH=/opt/ast:/opt/ast-extra

WORKDIR /workspace

