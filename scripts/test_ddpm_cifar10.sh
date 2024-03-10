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
path_1='models/experiments/cifar10/20240307-195731_9e95e'
path_2='models/experiments/cifar10/20240307-195733_7d01e'
path_3='models/experiments/cifar10/20240307-200155_9c240'
path_4='models/experiments/cifar10/20240307-200155_4cc5a'
path_5='models/experiments/cifar10/20240307-200400_2af47'
path_6='models/experiments/cifar10/20240307-200400_78df0'
path_7='models/experiments/cifar10/20240307-200740_ac79b'
path_8='models/experiments/cifar10/20240307-200740_bd0d2'
path_9='models/experiments/cifar10/20240307-200816_7dd64'
path_10='models/experiments/cifar10/20240307-201256_a4ce2'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# list of the chkpt_path strings to the trained DDPM models
ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0151.ckpt'
ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0139.ckpt'
ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0127.ckpt'
ddpm_path_4='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_4/checkpoints/ddpmv2-vae-epoch=999-loss=0.0173.ckpt'
ddpm_path_5='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_5/checkpoints/ddpmv2-vae-epoch=999-loss=0.0161.ckpt'
ddpm_path_6='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_6/checkpoints/ddpmv2-vae-epoch=999-loss=0.0133.ckpt'
ddpm_path_7='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_7/checkpoints/ddpmv2-vae-epoch=999-loss=0.0193.ckpt'
ddpm_path_8='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_8/checkpoints/ddpmv2-vae-epoch=999-loss=0.0150.ckpt'
ddpm_path_9='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_9/checkpoints/ddpmv2-vae-epoch=999-loss=0.0159.ckpt'
ddpm_path_10='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/seed_10/checkpoints/ddpmv2-vae-epoch=999-loss=0.0177.ckpt'
# create the list of chkpt_path strings
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results/cifar10/'

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode"
  done
done

## list of vae_chkpt_path strings to the trained TreeVAE models
#path_1='models/experiments/cifar10/20240229-100309_d92b3'
#path_2='models/experiments/cifar10/20240229-101730_bf3e0'
#path_3='models/experiments/cifar10/20240229-103928_3096f'
#path_4='models/experiments/cifar10/20240229-103928_290d7'
#path_5='models/experiments/cifar10/20240229-103928_5adb4'
#path_6='models/experiments/cifar10/20240229-103928_00878'
#path_7='models/experiments/cifar10/20240229-104852_dcbff'
#path_8='models/experiments/cifar10/20240229-105018_846e6'
#path_9='models/experiments/cifar10/20240229-105018_ef98c'
#path_10='models/experiments/cifar10/20240229-105018_deb67'