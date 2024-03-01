#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cifar10"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings
path_1='models/experiments/cifar10/'
path_2='models/experiments/cifar10/'
path_3='models/experiments/cifar10/'
path_4='models/experiments/cifar10/'
path_5='models/experiments/cifar10/'
path_6='models/experiments/cifar10/'
path_7='models/experiments/cifar10/'
path_8='models/experiments/cifar10/'
path_9='models/experiments/cifar10/'
path_10='models/experiments/cifar10/'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed_ddpm $seed"
done