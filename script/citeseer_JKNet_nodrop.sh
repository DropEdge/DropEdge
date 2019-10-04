#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 6 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.004 \
    --weight_decay 5e-05 \
    --early_stopping 400 \
    --sampling_percent 1 \
    --dropout 0.3 \
    --normalization AugNormAdj \
    --withloop \
    
