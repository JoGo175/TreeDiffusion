#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings
path_1='models/experiments/mnist/20240228-114539_b2acb'
path_2='models/experiments/mnist/20240228-115653_f566f'
path_3='models/experiments/mnist/20240228-120404_3363c'
path_4='models/experiments/mnist/20240228-121632_a8e0f'
path_5='models/experiments/mnist/20240228-122157_d5fc5'
path_6='models/experiments/mnist/20240228-122408_afb4c'
path_7='models/experiments/mnist/20240228-123049_4571f'
path_8='models/experiments/mnist/20240228-122932_ac9f8'
path_9='models/experiments/mnist/20240228-123127_d03f5'
path_10='models/experiments/mnist/20240228-130223_0a478'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed"
done
