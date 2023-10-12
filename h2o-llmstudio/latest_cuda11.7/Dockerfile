ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "tqdm>=4.65.0, <5.0.0" \
        "transformers==4.34.0" \
	"numpy>=1.23.2, <2.0.0" \
	"pandas>=2.1.0, <3.0.0" \
	"scikit-learn>=1.0.2, <2.0.0" \
	"boto3>=1.20.24, <2.0.0" \
	"SQLAlchemy>=1.4.36, <2.0.0" \
	"dill>=0.3.4, <0.4.0" \
	"pyarrow>=13.0.0" \
	"kaggle>=1.5.12, <2.0.0" \
	"coolname>=2.2.0, <3.0.0" \
	"bokeh>=3.2.1, <4.0.0" \
	"beautifulsoup4>=4.11.1, <5.0.0" \
	"sqlitedict==1.7.0" \
	"sentencepiece>=0.1.96, <0.2.0" \
	"sacrebleu==2.0.0" \
	"toml>=0.10.2, <0.11.0" \
	"pyyaml>=6.0.0, <7.0.0" \
	"protobuf==3.20.3" \
	"fastparquet>=2023.7.0" \
	"gputil>=1.4.0, <2.0.0" \
	"huggingface-hub==0.17.2" \
	"starlette>=0.31.0, <0.32.0" \
	"bitsandbytes==0.41.1" \
	"accelerate==0.23.0" \
	"openai>=0.28.0, <0.29.0" \
	"einops==0.6.1" \
	"datasets>=2.11.0, <3.0.0" \
	"neptune==1.6.3" \
	"Jinja2>=3.1.2, <4.0.0" \
	"tenacity>=8.2.2, <9.0.0" \
	"h2o-wave==0.26.2" \
	"tiktoken==0.5.1" \
	"hf-transfer==0.1.3" \
	"peft==0.5.0" \
	"azure-storage-file-datalake>=12.12.0" \
	"keyring==24.2.0"

WORKDIR /opt

RUN git clone https://github.com/h2oai/h2o-llmstudio.git

ENV PYTHONPATH=/opt/h2o-llmstudio

