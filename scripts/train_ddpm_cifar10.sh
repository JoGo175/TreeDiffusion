#!/bin/bash
#SBATCH --time=100:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cifar10"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/'

# Fully unconditional 
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds 
for seed in 1 2 3; do
  results_dir="${base_results_dir}fully_uncond/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond False"
done


# Conditioning on Leaf Reconstructions
# type = “form1”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds 
for seed in 1 2 3; do
  results_dir="${base_results_dir}cond_on_recons/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond False"
done


# Conditioning on Leaf Reconstructions + Leaf Index
# type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”

# loop over seeds 
for seed in 1 2 3; do
  results_dir="${base_results_dir}cond_on_recons_and_index/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
done


# Conditioning on Leaf Reconstructions + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “latent”

# loop over seeds
for seed in 1 2 3; do
  results_dir="${base_results_dir}cond_on_recons_and_emb/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal latent"
done


# Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “both”

# loop over seeds
for seed in 1 2 3; do
  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal both"
done