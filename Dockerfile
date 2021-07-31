FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    libgtk2.0-dev \
    libgl1-mesa-glx \
	git \
    wget \
	zsh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install dgl-cu101
RUN pip install -U git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install git+https://github.com/facebookresearch/detectron2.git@v0.1.3#egg=detectron2
RUN apt-get autoremove -y && apt-get autoclean -y

ADD ./models ./models
ADD ./utils ./utils
ADD ./weight ./weight
ADD ./examples ./examples
ADD ./*.py ./
ADD ./README.md ./