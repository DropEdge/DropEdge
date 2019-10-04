#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 2 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.009 \
    --weight_decay 0.001 \
    --early_stopping 400 \
    --sampling_percent 0.05 \
    --dropout 0.8 \
    --normalization BingGeNormAdj \
    --withloop \
    --withbn
