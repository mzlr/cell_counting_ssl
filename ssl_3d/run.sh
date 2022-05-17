#!/bin/bash
set -e

CONFIG_DIR=${1:-1}

OUTPUT_DIR=outputs/${CONFIG_DIR%%.*}

mkdir -p ${OUTPUT_DIR}
cp -f ${CONFIG_DIR} ${OUTPUT_DIR}/config.yaml

python train.py \
    --config_file ${OUTPUT_DIR}/config.yaml \
    OUTPUT_DIR ${OUTPUT_DIR} \
    2>&1 | tee -a ${OUTPUT_DIR}/log.txt
