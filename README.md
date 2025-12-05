# Pokemon Classification

A PyTorch-based image classification project that classifies Pokemon images using a four convolutional neural network architecture models.

## Dataset

The dataset is sourced from [Kaggle - Pokemon Classification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) and contains images of 151 different Pokemon species organized into training and testing sets.

## Instrucciones

Este código cuenta con cuatro posibles modelos a correr. Para eso se necesitan comandos concretos en la terminal para cada modelo.

1. resnet18
set MODEL_NAME=resnet18
set PRETRAINED=1
python src\train.py

2. efficientnet_b0
set MODEL_NAME=efficientnet_b0
set PRETRAINED=1
python src\train.py

3. mobilenet_v3_small
set MODEL_NAME=mobilenet_v3_small
set PRETRAINED=1
python src\train.py

4. tinyvgg (por defecto)
python src\train.py
