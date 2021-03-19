ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /usr -f && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN conda install pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch

RUN cd /opt && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd /opt/detectron2 && \
    git reset --hard 34acc94d862b8c31f382bee5013737184f444b74 && \
    python -m pip install --no-cache-dir .

RUN cd /opt && \
    git clone https://github.com/facebookresearch/mobile-vision.git && \
    cd /opt/mobile-vision && \
    git reset --hard af4dffe4662f13210c8fc2581a6d4e4c24b7e5c8 && \
    python -m pip install --no-cache-dir .

RUN cd /opt && \
    git clone https://github.com/facebookresearch/d2go.git && \
    cd /opt/d2go && \
    git reset --hard ec2e8ffffc8b59f4fabf7b1f3f31861eb7a4af70 && \
    python -m pip install --no-cache-dir .

COPY bash.bashrc /etc/bash.bashrc
#COPY d2go_train_coco.py /opt/d2go_ext/d2go_train_coco.py
#COPY d2go_train_coco /usr/bin/d2go_train_coco
