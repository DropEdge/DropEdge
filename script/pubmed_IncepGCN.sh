#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset pubmed \
    --type inceptiongcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 2 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.002 \
    --weight_decay 1e-05 \
    --early_stopping 400 \
    --sampling_percent 0.5 \
    --dropout 0.8 \
    --normalization BingGeNormAdj \
    --withloop \
    --withbn
