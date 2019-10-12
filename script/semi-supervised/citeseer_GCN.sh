#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 0 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --early_stopping 400 \
    --sampling_percent 0.2 \
    --dropout 0.3 \
    --normalization AugNormAdj --task_type semi \
     \
    
