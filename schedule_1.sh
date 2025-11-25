#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/train.py \
    model._target_=src.models.components.mobile_net.MobileNet \
    experiment=surface_mixup_nce_rce \
    trainer.device_id=${GPU_ID}