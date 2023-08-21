FROM waikatodatamining/pytorch-huggingface-transformers:4.31.0_cuda11.7

RUN pip install --no-cache-dir git+https://github.com/rmihaylov/falcontune.git@6bd029e5a89f58c4eea9056ee1c86127e1200876 && \
    pip install --no-cache-dir wandb==0.15.3 sentencepiece==0.1.99

RUN pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/falcontune/
COPY falcon* /usr/local/bin/

ENV PYTHONPATH=/opt/falcontune
