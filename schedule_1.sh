#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/sweep.py \
    model._target_=src.models.components.efficient_net.EfficientNet \
    experiment=surface_sgd \
    hparams_search=sgd \
    seed=21 \
    trainer.device_id=${GPU_ID}