#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="celeba"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"


# list of vae_chkpt_path strings to the trained TreeVAE models
path_1='models/experiments/celeba/20240918-032531_103db'
path_2='models/experiments/celeba/20240918-032554_0d008'
path_3='models/experiments/celeba/20240918-032615_143b9'
path_4='models/experiments/celeba/20240918-032635_0206c'
path_5='models/experiments/celeba/20240918-032751_46c3e'
path_6='models/experiments/celeba/20240918-032749_c73e9'
path_7='models/experiments/celeba/20240918-032937_80cf8'
path_8='models/experiments/celeba/20240918-032915_b4bde'
path_9='models/experiments/celeba/20240918-033153_8f6e7'
path_10='models/experiments/celeba/20240918-033153_3d955'
vae_path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/celeba/'

# Conditioning on Path
# type = ?uncond?, z_cond = True, z_dim = 1024, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0146.ckpt"
ddpm_path_2="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0142.ckpt"
ddpm_path_3="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0143.ckpt"
ddpm_path_4="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0083.ckpt"
ddpm_path_5="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed5-epoch=994-loss=0.0160.ckpt"
ddpm_path_6="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0142.ckpt"
ddpm_path_7="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0113.ckpt"
ddpm_path_8="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0168.ckpt"
ddpm_path_9="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0094.ckpt"
ddpm_path_10="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0135.ckpt"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_path/ddim/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
      --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
  done
done