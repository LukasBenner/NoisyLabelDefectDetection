#!/bin/sh

GPU_ID=1

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/sweep.py \
    experiment=loss_ngce \
    hparams_search=loss_ngce \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_nmae \
    hparams_search=loss_nmae \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_nrce \
    hparams_search=loss_nrce \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_sce \
    hparams_search=loss_sce \
    trainer.devices=[$GPU_ID]