#!/bin/bash
#SBATCH --time=36:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"


for mode in 'vae_recons' 'vae_samples'; do
  sbatch --time=36:00:00 --mem-per-cpu=15G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python vae_generations.py --config_name $dataset --mode $mode"
done
