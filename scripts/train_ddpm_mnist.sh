#!/bin/bash
#SBATCH --time=168:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

# list of vae_chkpt_path strings to the trained TreeVAE models
path_1='models/experiments/mnist/20240906-142239_b81c6'
path_2='models/experiments/mnist/20240906-142242_58da9'
path_3='models/experiments/mnist/20240906-142349_7bbc5'
path_4='models/experiments/mnist/20240906-142601_9adee'
path_5='models/experiments/mnist/20240906-142544_f963b'
path_6='models/experiments/mnist/20240906-142555_fdc8e'
path_7='models/experiments/mnist/20240906-142605_1e210'
path_8='models/experiments/mnist/20240906-142529_6f0b3'
path_9='models/experiments/mnist/20240906-142703_1bf93'
path_10='models/experiments/mnist/20240906-142703_88adf'
# create the list of vae_chkpt_path strings
path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/mnist/'

# Fully unconditional
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}fully_uncond/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond False"
done



# Conditioning on Leaf Reconstructions
# type = “form1”, z_cond = False, z_dim = None, z_signal = None

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond False"
done


# Conditioning on Leaf Reconstructions + Leaf Index
# type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_index/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
done


# Conditioning on Leaf Reconstructions + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “latent”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_emb/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal latent"
done


# Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “both”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal both"
done


# Conditioning on Leaf Index + Leaf Embeddings
# type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “both”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_index_and_emb/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal both"
done


# Conditioning on Leaf Reconstructions + Path
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “path”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_path/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal path"
done


# Conditioning on Path
# type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “path”

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_path/seed_${seed}/"
  # run the job
  sbatch --time=168:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python train_ddpm.py --config_name $dataset --vae_chkpt_path ${path_list[$seed-1]} --results_dir $results_dir --seed $seed --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
done
