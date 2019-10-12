#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type inceptiongcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 6 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.002 \
    --weight_decay 0.005 \
    --early_stopping 400 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization BingGeNormAdj \
    --withloop \
    
