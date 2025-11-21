#!/bin/sh

GPU_ID=0

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include

python src/train.py \
    model._target_=src.models.components.mobile_net.MobileNet \
    experiment=surface_baseline \
    num_runs=10 \
    trainer.device_id=${GPU_ID}


python src/train.py \
    model._target_=src.models.components.efficient_net.EfficientNet \
    experiment=surface_baseline \
    num_runs=10 \
    trainer.device_id=${GPU_ID}
