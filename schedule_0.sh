#!/bin/sh

GPU_ID=0

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/train.py \
    experiment=surface_sgd_mixup \
    trainer.device_id=${GPU_ID}

python src/train.py \
    experiment=surface_sgd_nce_rce \
    trainer.device_id=${GPU_ID}
