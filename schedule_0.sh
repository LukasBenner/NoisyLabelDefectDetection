#!/bin/sh

GPU_ID=0

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/train.py \
    experiment=loss_ce_mobilenet \
    trainer.devices=[$GPU_ID]

python src/train.py \
    experiment=loss_nce_and_rce_mobilenet \
    trainer.devices=[$GPU_ID]