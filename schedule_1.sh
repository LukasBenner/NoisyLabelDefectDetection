#!/bin/sh

python src/train_multiple_runs.py experiment=surface_baseline_improved trainer.devices=[1] +num_runs=10