#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset"


# path for vae_chkpt_path strings to the trained TreeVAE models
path='models/experiments/cifar10/20240506-143804_675eb'

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/'

# loop over seeds and vae_chkpt_path
for seed in 42; do
  for epoch in 100 500 1000 2000 5000; do
    results_dir="${base_results_dir}seed_${seed}/epoch_${epoch}/"
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path $path --results_dir $results_dir --seed $seed --epochs $epoch"
  done
done


