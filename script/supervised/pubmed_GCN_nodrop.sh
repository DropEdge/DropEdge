#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset pubmed \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 2 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.01 \
    --weight_decay 0.001 \
    --early_stopping 400 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization BingGeNormAdj \
    --withloop \
    --withbn
