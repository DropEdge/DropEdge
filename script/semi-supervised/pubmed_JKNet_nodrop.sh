#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset pubmed \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 30 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.001 \
    --weight_decay 0.005 \
    --early_stopping 400 \
    --sampling_percent 1 \
    --dropout 0.8 \
    --normalization AugNormAdj --task_type semi \
     \
    
