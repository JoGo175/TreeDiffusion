#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="fmnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings to the trained TreeVAE models
path_1='models/experiments/fmnist/20240301-204616_da161'
path_2='models/experiments/fmnist/20240301-214631_426ff'
path_3='models/experiments/fmnist/20240301-214630_72678'
path_4='models/experiments/fmnist/20240301-214704_24277'
path_5='models/experiments/fmnist/20240301-214949_7e0c7'
path_6='models/experiments/fmnist/20240301-214949_ef88e'
path_7='models/experiments/fmnist/20240301-214949_f6999'
path_8='models/experiments/fmnist/20240301-214617_40bff'
path_9='models/experiments/fmnist/20240301-214617_cdc1b'
path_10='models/experiments/fmnist/20240301-214617_8b3a7'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/fmnist/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed"
done

## list of vae_chkpt_path strings to the trained TreeVAE models
#path_1='models/experiments/fmnist/20240301-060407_1bdf0'
#path_2='models/experiments/fmnist/20240301-060408_a24b4'
#path_3='models/experiments/fmnist/20240301-060423_4e2bb'
#path_4='models/experiments/fmnist/20240301-060921_6ba52'
#path_5='models/experiments/fmnist/20240301-060951_6feb0'
#path_6='models/experiments/fmnist/20240301-061104_19a4f'
#path_7='models/experiments/fmnist/20240302-152702_45499'
#path_8='models/experiments/fmnist/20240301-061325_54312'
#path_9='models/experiments/fmnist/20240301-061325_ff0e8'
#path_10='models/experiments/fmnist/20240301-061348_26b65'