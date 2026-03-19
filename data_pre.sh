#!/bin/bash

DATA_DIR="./data/facebook_test"
GPU_ID=7
SIMILARITY_THRESHOLD=0.7
VIRTUAL_GROUP_WINDOW=50
MAX_VOCAB_SIZE=2000
LIMIT=300

CUDA_VISIBLE_DEVICES=$GPU_ID python preprocess_facebook_bge.py \
    "$DATA_DIR" \
    --similarity_threshold $SIMILARITY_THRESHOLD \
    --virtual_group_window $VIRTUAL_GROUP_WINDOW \
    --max_vocab_size $MAX_VOCAB_SIZE \
    --limit $LIMIT
