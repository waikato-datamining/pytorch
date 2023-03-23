FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget g++ ca-certificates git-lfs build-essential python3.7 python3.7-dev python3-snappy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir datasets==2.10.1 loguru==0.6.0 netifaces==0.11.0 transformers==4.21.1 wandb==0.13.10 zstandard==0.20.0 && \
    pip3 install --no-cache-dir cupy==10.4.0 faiss-gpu==1.7.2 fastparquet==0.5.0 pyarrow==8.0.0 python-snappy==0.6.1

WORKDIR /opt
RUN git clone https://github.com/togethercomputer/OpenChatKit.git && \
    cd OpenChatKit && \
    git reset --hard 700cbf27bdfca0f2173d906ac25867e9bb51c451

COPY bash.bashrc /etc/bash.bashrc

ENV PYTHONPATH=/opt/OpenChatKit

