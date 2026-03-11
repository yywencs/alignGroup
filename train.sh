#!/bin/bash

DATASET="facebook_stopwords"
CHECKPOINT_DIR="./checkpoints/${DATASET}"

CUDA_VISIBLE_DEVICE=0 python main.py \
    --dataset "$DATASET" \
    --learning_rate 0.001 \
    --cl_weight 0.01 \
    --temp 0.2 \
    --epoch 100 \
    --batch_size 512 \
    --layers 4 \
    --emb_dim 32 \
    --num_negatives 8 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --patience 4 \
    --num_negatives 16