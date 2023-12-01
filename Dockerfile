FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt update && \
    apt install -y python3 python3-pip && \
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    ln -s /usr/bin/python3 /usr/bin/python && pip install onnxruntime
