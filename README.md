# Simple-CNN-Example

## Introduction
This repository aims at introducing how to train deep leaerning classification models with Pytorch, 
export to onnx and use it with onnxruntime taking MNIST dataset, which is famous for handwriting digit image, as an example.
Generally, CNN model accepts 3channels(RGB) but MNIST has one channel. To deal with this, Custom MNIST Dataset class returns 3channels tensor inheriting "torchvision.dataset.MNIST" class.

## Prerequisites
* Docker
* Docker compose
* docker login nvcr.io
* dGPU (Recommended)

## How to train
### Train with dGPU
```bash
docker compose -f docker-compose-gpu.yaml up -d
docker exec -it mnist_train /bin/bash
python train.py
```

### Train with cpu
```bash
docker compose -f docker-compose.yaml up -d
docker exec -it mnist_train /bin/bash
python train.py
```

## Export model from pytorch to onnx
After training, run command bellow.
```bash
python export.py
```

## Run onnx with onnxruntime
```bash
python check_onnx_inferenc.py
```
The code above choose 3 sample images from MNIST dataset, infer them and show results of inference of pytorch model and onnx model.