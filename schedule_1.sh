#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/train.py \
    experiment=surface_baseline \
    trainer.device_id=${GPU_ID}

python src/train.py \
    experiment=surface_scheduler \
    trainer.device_id=${GPU_ID}

python src/train.py \
    experiment=surface_sgd \
    trainer.device_id=${GPU_ID}