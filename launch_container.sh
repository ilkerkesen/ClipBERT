#!/bin/bash

REPO_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONTAINER_PATH=$1
PATH_TO_STORAGE=$2
JSON_DIR=$3
OUTPUT_DIR=$4
DATA=$PATH_TO_STORAGE
QUVA_DIR=${QUVA_DIR:-""}
SOMETHING_SOMETHING_DIR=${SOMETHING_SOMETHING_DIR:-""}
export CONFIG_PATH=/clipbert/src/configs/msrvtt_ret_base_resnet50.json

singularity shell \
    --bind $REPO_DIR:/clipbert \
    --bind $DATA/pretrained:/pretrain \
    --bind $DATA/txt_db:/txt \
    --bind $DATA/vis_db:/img \
    --bind $OUTPUT_DIR:/storage \
    --bind $JSON_DIR:/annotations \
    --bind $QUVA_DIR:/quva \
    --bind $SOMETHING_SOMETHING_DIR:/something \
    --nv $CONTAINER_PATH