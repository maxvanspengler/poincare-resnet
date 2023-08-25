#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poincare_resnet

declare -a models=(
    "hyperbolic-8-16-32-resnet-32"
    "euclidean-8-16-32-resnet-32"
    "euclideanwhypclass-8-16-32-resnet-32"
)

declare -a epsilons=(
    "0.00314"
    "0.00627"
    "0.00941"
    "0.01255"
)

for model in "${models[@]}"; do
    echo $model
    for epsilon in "${epsilons[@]}"; do
        echo $epsilon
        python -m adversarial_attacks \
            $model cifar10 \
            -e $epsilon \
            --batch-size 128
    done
done
