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
# type = ?uncond?, z_cond = True, z_dim = 64, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0173.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons/ddim/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample_all_leaves'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
      --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond False"
  done
done
