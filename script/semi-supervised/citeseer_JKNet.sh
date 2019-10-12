#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 14 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.006 \
    --weight_decay 0.001 \
    --early_stopping 400 \
    --sampling_percent 0.5 \
    --dropout 0.8 \
    --normalization AugNormAdj --task_type semi \
    --withloop \
    
