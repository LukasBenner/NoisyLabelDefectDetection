#!/bin/sh

GPU_ID=0

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/train_lightning.py \
    experiment=surface_baseline \
    trainer.devices=[$GPU_ID]

python src/train_lightning.py \
    experiment=surface_scheduler \
    trainer.devices=[$GPU_ID]

python src/train_lightning.py \
    experiment=surface_sgd \
    trainer.devices=[$GPU_ID]