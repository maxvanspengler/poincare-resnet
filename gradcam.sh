#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poincare_resnet

python -m gradcam \
    8-16-32-resnet-32 \
    cifar10 \
    -e
