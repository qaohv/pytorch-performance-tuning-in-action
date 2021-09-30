# Pytorch performance tuning in action

Code for [article](https://medium.com/deelvin-machine-learning/pytorch-performance-tuning-in-action-7c4d065d4278) PyTorch performance tuning in action

## Requirements

1. NVIDIA driver == 455.32.00
2. Docker >= 19.03
3. nvidia-container-toolkit
4. make

## Dataset

Download [dataset](https://www.kaggle.com/tapakah68/supervisely-filtered-segmentation-person-dataset) and put in data folder

## Build docker image

```
docker build -t experiment .
```

## Runnining

#### To run particular experiment use following command:

```
make expN
```

where N is number of experiment

#### To run all experiments use following command:

```
make all
```
