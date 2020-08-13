ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# install basics
RUN apt-get update && \
    apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install warp-CTC
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/SeanNaren/warp-ctc.git /warp-ctc && \
    cd /warp-ctc && \
    git reset --hard 6118f8917c33724d5d31cde338362ad5b55a8cd8 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd /warp-ctc && \
    cd pytorch_binding && \
    python setup.py install

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git /ctcdecode && \
    cd /ctcdecode && \
    git reset --hard 958975c269123dbe705e947289e8ccb3855bffcd && \
    pip install --no-cache-dir .

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git /apex && \
    cd /apex && \
    git reset --hard 5d9b5cbc2fb0e0e525c067723d66f1827440c7a8 && \
    pip install --no-cache-dir .

# install deepspeech.pytorch
RUN git clone https://github.com/SeanNaren/deepspeech.pytorch.git /deepspeech.pytorch && \
    cd /deepspeech.pytorch && \
    git reset --hard 6518f6f286776cab92238e8b235a7fc37852ce70 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

COPY bash.bashrc /etc/bash.bashrc
COPY ds2_train /usr/bin/ds2_train
COPY ds2_transcribe /usr/bin/ds2_transcribe
