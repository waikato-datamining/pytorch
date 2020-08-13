ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY bash.bashrc /etc/bash.bashrc

RUN conda install cython -y && \
    conda clean --all && \
    git clone git clone https://github.com/ksanjeevan/crnn-audio-classification.git /crnnaclass && \
    cd /crnnaclass && \
    git reset --hard f09449b87ed61d28b82931c7055d033a7b69160b && \
    git clone https://github.com/keunwoochoi/torchaudio-contrib /torchaudio-contrib && \
    pip install --no-cache-dir -e /torchaudio-contrib && \
    pip install --no-cache-dir SoundFile && \
    pip install --no-cache-dir git+https://github.com/ksanjeevan/torchparse.git && \
    mkdir -m 0777 /.cache

COPY crnnaclass /usr/bin/crnnaclass
COPY bash.bashrc /etc/bash.bashrc

ENV PYTHONPATH=/crnnaclass

WORKDIR /crnnaclass

