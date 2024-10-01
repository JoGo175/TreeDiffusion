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
path_1='models/experiments/fmnist/20240906-144022_3c565'
path_2='models/experiments/fmnist/20240906-144229_49fab'
path_3='models/experiments/fmnist/20240906-145256_75a75'
path_4='models/experiments/fmnist/20240906-145908_8be88'
path_5='models/experiments/fmnist/20240906-150330_85017'
path_6='models/experiments/fmnist/20240906-151528_9aa30'
path_7='models/experiments/fmnist/20240906-152336_e1502'
path_8='models/experiments/fmnist/20240906-152742_bc919'
path_9='models/experiments/fmnist/20240906-153548_b01b6'
path_10='models/experiments/fmnist/20240906-153655_e8a5c'
# create the list of vae_chkpt_path strings
vae_path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/fmnist/'

# Conditioning on Path
# type = ?uncond?, z_cond = True, z_dim = 1024, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0123.ckpt"
ddpm_path_2="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed2-epoch=926-loss=0.0143.ckpt"
ddpm_path_3="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed3-epoch=988-loss=0.0144.ckpt"
ddpm_path_4="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed4-epoch=988-loss=0.0141.ckpt"
ddpm_path_5="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed5-epoch=990-loss=0.0084.ckpt"
ddpm_path_6="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed6-epoch=994-loss=0.0144.ckpt"
ddpm_path_7="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0122.ckpt"
ddpm_path_8="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0126.ckpt"
ddpm_path_9="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0181.ckpt"
ddpm_path_10="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed10-epoch=994-loss=0.0171.ckpt"
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