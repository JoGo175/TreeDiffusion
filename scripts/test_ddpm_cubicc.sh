#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cubicc"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"


# list of vae_chkpt_path strings to the trained TreeVAE models
path_1='models/experiments/cubicc/20240923-013355_8ddcc'
path_2='models/experiments/cubicc/20240923-013355_276bd'
path_3='models/experiments/cubicc/20240923-013355_2538a'
path_4='models/experiments/cubicc/20240923-013355_6236b'
path_5='models/experiments/cubicc/20240923-013355_fc06f'
path_6='models/experiments/cubicc/20240923-013355_434df'
path_7='models/experiments/cubicc/20240923-013355_61ff6'
path_8='models/experiments/cubicc/20240923-013355_ac769'
path_9='models/experiments/cubicc/20240923-013340_34724'
path_10='models/experiments/cubicc/20240923-013340_0e296'
vae_path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/cubicc/'


# Cond on Recons
# type = “form1”, z_cond = False, z_dim = None, z_signal = None

ddpm_path_1="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-new_vae_seed1-epoch=499-loss=0.0067.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons/ddim/new_seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample_all_leaves' 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond False"
  done
done



## Conditioning on Path
## type = ?uncond?, z_cond = True, z_dim = 1024, z_signal = ?path?
#
#ddpm_path_1="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed1-epoch=499-loss=0.0063.ckpt"
#ddpm_path_2="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed2-epoch=499-loss=0.0032.ckpt"
#ddpm_path_3="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed3-epoch=499-loss=0.0006.ckpt"
#ddpm_path_4="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed4-epoch=499-loss=0.0020.ckpt"
#ddpm_path_5="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed5-epoch=499-loss=0.0088.ckpt"
#ddpm_path_6="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed6-epoch=499-loss=0.0010.ckpt"
#ddpm_path_7="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed7-epoch=499-loss=0.0015.ckpt"
#ddpm_path_8="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed8-epoch=499-loss=0.0053.ckpt"
#ddpm_path_9="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed9-epoch=499-loss=0.0014.ckpt"
#ddpm_path_10="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed10-epoch=499-loss=0.0030.ckpt"
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)
#
## loop over seeds
#for seed in 1; do
#  results_dir="${base_results_dir}cond_on_path/ddim/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
#      --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
#  done
#done