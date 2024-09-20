#!/bin/bash
#SBATCH --time=168:00:00
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
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/fmnist/'

## Fully unconditional
## type = “uncond”, z_cond = False, z_dim = None, z_signal = None
#
## loop over seeds
#for seed in 1 2 3 4 5 6 7 8 9 10; do
#  results_dir="${base_results_dir}fully_uncond/"
#  chkpt_prefix="vae_seed${seed}"
#  # run the job
#  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond False"
#done
#
#
#
## Conditioning on Leaf Reconstructions
## type = “form1”, z_cond = False, z_dim = None, z_signal = None
#
## loop over seeds
#for seed in 1 2 3 4 5 6 7 8 9 10; do
#  results_dir="${base_results_dir}cond_on_recons/"
#  chkpt_prefix="vae_seed${seed}"
#  # run the job
#  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond False"
#done
#
#
## Conditioning on Leaf Reconstructions + Leaf Index
## type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”
#
## loop over seeds
#for seed in 1 2 3 4 5 6 7 8 9 10; do
#  results_dir="${base_results_dir}cond_on_recons_and_index/"
#  chkpt_prefix="vae_seed${seed}"
#  # run the job
#  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
#done


# Conditioning on Leaf Reconstructions + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 64, z_signal = “latent”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 64 --z_signal latent"
done


# Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 64, z_signal = “both”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 64 --z_signal both"
done


# Conditioning on Leaf Index + Leaf Embeddings
# type = “uncond”, z_cond = True, z_dim = 64, z_signal = “both”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_index_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 64 --z_signal both"
done


# Conditioning on Leaf Reconstructions + Path
# type = “form1”, z_cond = True, z_dim = 64, z_signal = “path”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_path/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 64 --z_signal path"
done


# Conditioning on Path
# type = “uncond”, z_cond = True, z_dim = 64, z_signal = “path”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_path/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 64 --z_signal path"
done
