#!/bin/bash
#SBATCH --time=36:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae

dataset="cifar10bash sc"

O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"


# MNIST
if [ "$dataset" == "mnist" ]; then
    O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"
    sbatch --time=36:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python classifier_pretraining/mnist_clf_training.py"
fi

# FashionMNIST
if [ "$dataset" == "fmnist" ]; then
    O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"
    sbatch --time=36:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python classifier_pretraining/fmnist_clf_training.py"
fi

# CIFAR10
if [ "$dataset" == "cifar10" ]; then
    O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"
    sbatch --time=36:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python classifier_pretraining/cifar10_clf_training.py"
fi