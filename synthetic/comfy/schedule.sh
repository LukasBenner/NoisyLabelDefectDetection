#!/bin/bash

# Step 1: Generate images with white background (for compositing later)
python generate.py \
    --server https://mxxm9fo4b4ogdd-8188.proxy.runpod.net \
    --template workflow_plain.json --timeout 120 \
    --prompt_template_file prompt_composite.txt \
    --defect_name missing_part --defect_block_file ./defects/missing_part.txt \
    --out_base ./output/pretrained/raw --count 220

# Step 2: Generate images with plain backgrounds (no compositing needed)
python generate.py \
    --server https://mxxm9fo4b4ogdd-8188.proxy.runpod.net \
    --template workflow_plain.json --timeout 120 \
    --prompt_template_file prompt_plain.txt \
    --defect_name missing_part --defect_block_file ./defects/missing_part.txt \
    --out_base ./output/pretrained/plain --count 250

# Step 3: Composite the white-background images onto real backgrounds (runs offline)
python composite.py \
    --input_dir ./output/pretrained/raw/missing_part \
    --background_dir ./backgrounds \
    --out_dir ./output/pretrained/composite/missing_part
