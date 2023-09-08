FROM waikatodatamining/pytorch-huggingface-transformers:4.7.0_cuda11.1

RUN pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/finetune-gpt2xl/
COPY run_clm.py /opt/finetune-gpt2xl/
COPY text2csv.py /opt/finetune-gpt2xl/
COPY ds_config*.json /opt/finetune-gpt2xl/
COPY gpt* /usr/local/bin/
COPY train_clm /usr/local/bin/

ENV PYTHONPATH=/opt/finetune-gpt2xl
