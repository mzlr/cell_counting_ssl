# Self-supervised learning for cell counting
This repo contains the source code for 2D & 3D image-based cell counting with self-supervised learning.

## Prerequisites
- [Docker](https://docs.docker.com/engine/install/)
- Docker Container: [20.06-tf1-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags) `docker pull nvcr.io/nvidia/tensorflow:20.06-tf1-py3`

## Usage
Inside the docker container, run the following code for different data
### 2D
```sh
cd ssl_2d
bash run.sh config.yaml
```
### 3D
```sh
cd ssl_3d
bash run.sh config.yaml
```
