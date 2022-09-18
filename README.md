# Self-supervised learning for cell counting
This repo contains the source code for the paper, **Cell counting with inverse distance kernel and self-supervised learning** ([link](https://link.springer.com/chapter/10.1007/978-3-031-16961-8_1)).

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
## Data
Pre-processed data used in the paper are also provided for convenience.
Create the `data` folder in the both the 2D & 3D folder and download the files from [Google Drive](https://drive.google.com/drive/folders/1WUG4VSZ4vktmkTB549Pnpv1pTAm0cFVw?usp=sharing).  The structure is as follows:
```
cell_counting_ssl
  |-ssl_2d
  |  |-data
  |  |  |-VGG
  |  |  |  |-VGG.hdf5
  ...
  |-ssl_3d
  |  |-data
  |  |  |-STN
  |  |  |  |-075_fid.hdf5
  |  |  |  |-unlabeled.hdf5
  ...
```

# Interactive labeling tool for dot annotations 
Our tool is based on [Segmentor](https://github.com/RENCI/Segmentor). Please refer to this [wiki](https://github.com/RENCI/Segmentor/wiki/Dot-Annotation) for details.
