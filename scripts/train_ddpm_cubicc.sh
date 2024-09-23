#!/bin/bash
#SBATCH --time=168:00:00
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
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/cubicc/'

# Fully unconditional
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}fully_uncond/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond False"
done



# Conditioning on Leaf Reconstructions
# type = “form1”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond False"
done


# Conditioning on Leaf Reconstructions + Leaf Index
# type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_index/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
done


# Conditioning on Leaf Reconstructions + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “latent”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal latent"
done


# Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “both”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal both"
done


# Conditioning on Leaf Index + Leaf Embeddings
# type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “both”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_index_and_emb/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal both"
done


# Conditioning on Leaf Reconstructions + Path
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “path”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_path/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal path"
done


# Conditioning on Path
# type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “path”

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_path/"
  chkpt_prefix="vae_seed${seed}"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:rtx4090:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
done
