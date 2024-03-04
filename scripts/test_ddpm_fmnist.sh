#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings to the trained TreeVAE models
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

# list of the chkpt_path strings to the trained DDPM models
ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0074.ckpt'
ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0082.ckpt'
ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0091.ckpt'
ddpm_path_4='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_4/checkpoints/ddpmv2-vae-epoch=999-loss=0.0077.ckpt'
ddpm_path_5='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_5/checkpoints/ddpmv2-vae-epoch=999-loss=0.0092.ckpt'
ddpm_path_6='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_6/checkpoints/ddpmv2-vae-epoch=999-loss=0.0093.ckpt'
ddpm_path_7='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_7/checkpoints/ddpmv2-vae-epoch=999-loss=0.0086.ckpt'
ddpm_path_8='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_8/checkpoints/ddpmv2-vae-epoch=999-loss=0.0083.ckpt'
ddpm_path_9='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_9/checkpoints/ddpmv2-vae-epoch=999-loss=0.0102.ckpt'
ddpm_path_10='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/seed_10/checkpoints/ddpmv2-vae-epoch=999-loss=0.0123.ckpt'
# create the list of chkpt_path strings
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/mnist/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'sample_all_leaves' 'recons' 'recons_all_leaves'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --ddpm_chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode"
  done
done