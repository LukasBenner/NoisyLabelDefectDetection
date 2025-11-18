#!/bin/sh

python src/train_kfold.py experiment=surface_baseline trainer.devices=[0] data.n_splits=5