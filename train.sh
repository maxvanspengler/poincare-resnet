#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poincare_resnet

python -m train \
    euclidean-8-16-32-resnet-20 \
    cifar100 \
    -e 100 \
    -s \
    --opt=adam \
    --lr=0.001 \
    --weight-decay=1e-4
