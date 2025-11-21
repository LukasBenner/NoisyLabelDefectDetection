#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include

python src/train_simple.py \
    model._target_=src.models.components.efficient_net.EfficientNet \
    num_runs=10 \
    trainer.device_id=${GPU_ID}