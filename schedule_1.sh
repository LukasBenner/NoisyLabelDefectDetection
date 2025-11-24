#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include

python src/sweep.py \
    model._target_=src.models.components.mobile_net.MobileNet \
    experiment=surface_mixup \
    hparams_search=mixup \
    n_trials=20 \
    trainer.device_id=${GPU_ID}

python src/train.py \
    model._target_=src.models.components.efficient_net.EfficientNet \
    experiment=surface_mixup \
    hparams_search=mixup \
    n_trials=20 \
    trainer.device_id=${GPU_ID}