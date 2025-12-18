#!/bin/sh

GPU_ID=0

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/cuda/include
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

python src/sweep.py \
    experiment=loss_ce \
    hparams_search=loss_ce \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_gce_and_mae \
    hparams_search=loss_gce_and_mae \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_gce_and_nce \
    hparams_search=loss_gce_and_nce \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_gce_and_rce \
    hparams_search=loss_gce_and_rce \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_nce_and_mae \
    hparams_search=loss_nce_and_mae \
    trainer.devices=[$GPU_ID]

python src/sweep.py \
    experiment=loss_nce_and_rce \
    hparams_search=loss_nce_and_rce \
    trainer.devices=[$GPU_ID]