#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cifar10"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# loop over the datasets
for dataset in "mnist" "fmnist" "cifar10" "celeba" "cubicc"; do
    # loop over the seeds
    for seed in 0 1 2 3 4 5 6 7 8 9 10; do
        O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}_${seed}.out"
        sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_vanilla_vae.py --config_name $dataset --seed $seed --chkpt_prefix 'vae_${seed}'"
    done
done
