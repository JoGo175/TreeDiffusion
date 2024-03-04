#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cifar10"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings to the trained TreeVAE models
path_1='models/experiments/cifar10/20240229-100309_d92b3'
path_2='models/experiments/cifar10/20240229-101730_bf3e0'
path_3='models/experiments/cifar10/20240229-103928_3096f'
path_4='models/experiments/cifar10/20240229-103928_290d7'
path_5='models/experiments/cifar10/20240229-103928_5adb4'
path_6='models/experiments/cifar10/20240229-103928_00878'
path_7='models/experiments/cifar10/20240229-104852_dcbff'
path_8='models/experiments/cifar10/20240229-105018_846e6'
path_9='models/experiments/cifar10/20240229-105018_ef98c'
path_10='models/experiments/cifar10/20240229-105018_deb67'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed"
done