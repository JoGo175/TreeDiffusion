#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cubicc"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python cubicc_FID.py --config_name $dataset"
