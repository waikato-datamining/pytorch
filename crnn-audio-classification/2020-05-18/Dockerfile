ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get install -y --no-install-recommends libopus0 libsndfile1 && \
    apt-get install -y --no-install-recommends ffmpeg sox libsox-fmt-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY bash.bashrc /etc/bash.bashrc

RUN conda install cython -y && \
    conda clean --all && \
    git clone https://github.com/ksanjeevan/crnn-audio-classification.git /crnnaclass && \
    cd /crnnaclass && \
    git reset --hard f09449b87ed61d28b82931c7055d033a7b69160b && \
    pip install --no-cache-dir SoundFile matplotlib pandas torchaudio && \
    git clone https://github.com/keunwoochoi/torchaudio-contrib /torchaudio-contrib && \
    pip install --no-cache-dir -e /torchaudio-contrib && \
    pip install --no-cache-dir git+https://github.com/ksanjeevan/torchparse.git && \
    pip install --no-cache-dir tensorboardX tensorboard tensorflow && \
    mkdir -m 0777 /.cache

COPY crnnaclass /usr/bin/crnnaclass
COPY bash.bashrc /etc/bash.bashrc

ENV PYTHONPATH=/crnnaclass

WORKDIR /crnnaclass

