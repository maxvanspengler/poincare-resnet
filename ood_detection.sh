#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poincare_resnet

declare -a datasets=(
    "cifar10"
    "cifar100"
)

declare -a models=(
    "hyperbolic-8-16-32-resnet-20"
    "euclidean-8-16-32-resnet-20"
    "euclideanwhypclass-8-16-32-resnet-20"
    "hyperbolic-8-16-32-resnet-32"
    "euclidean-8-16-32-resnet-32"
    "euclideanwhypclass-8-16-32-resnet-32"
)

for dataset in "${datasets[@]}"; do
    echo $dataset
    for model in "${models[@]}"; do
        echo $model
        python -m ood_detection \
            --model $model \
            --dataset $dataset \
            --num_to_avg 10
    done
done
