# Pytorch performance tuning in action

Code for article: [TODO]

## Requirements

1. NVIDIA driver == 455.32.00
2. Docker >= 19.03
3. nvidia-container-toolkit
4. make

## Dataset

https://www.kaggle.com/tapakah68/supervisely-filtered-segmentation-person-dataset

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
