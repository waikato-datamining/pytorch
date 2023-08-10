ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch-huggingface_transformers:4.31.0_cuda11.7

RUN pip install --no-cache-dir git+https://github.com/rmihaylov/falcontune.git@6bd029e5a89f58c4eea9056ee1c86127e1200876 && \
    pip install --no-cache-dir wandb==0.15.3 sentencepiece==0.1.99

COPY bash.bashrc /etc/bash.bashrc
