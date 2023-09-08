FROM waikatodatamining/pytorch-huggingface-transformers:4.31.0_cuda11.7

RUN pip install --no-cache-dir sacrebleu==2.3.1 evaluate==0.4.0

RUN pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY run*.py /opt/translate/
COPY predict*.py /opt/translate/
COPY translation_* /usr/local/bin/

ENV PYTHONPATH=/opt/translate
