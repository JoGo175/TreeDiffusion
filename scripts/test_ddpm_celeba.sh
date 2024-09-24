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


# Fully unconditional
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

ddpm_path_1="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0034.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}fully_uncond/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond'"
  done
done



# Conditioning on Leaf Reconstructions
# type = “form1”, z_cond = False, z_dim = None, z_signal = None

ddpm_path_1="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0082.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond False"
  done
done



# Conditioning on Leaf Reconstructions + Leaf Index
# type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”

ddpm_path_1="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0082.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_index/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
  done
done



# Conditioning on Leaf Reconstructions + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “latent”

ddpm_path_1="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0084.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_emb/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal latent"
  done
done




# Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
# type = “form1”, z_cond = True, z_dim = 1024, z_signal = “both”

ddpm_path_1="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0081.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal both"
  done
done


# Conditioning on Leaf Index + Leaf Embeddings
# type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “both”

ddpm_path_1="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=249-loss=0.0085.ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds and vae_chkpt_path
for seed in 1; do
  results_dir="${base_results_dir}cond_on_index_and_emb/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal both"
  done
done


# Conditioning on Leaf Reconstructions + Path
# type = ?form1?, z_cond = True, z_dim = 1024, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed1-epoch=193-loss=0.0055 (copy 1).ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_recons_and_path/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
      --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal path"
  done
done


# Conditioning on Path
# type = ?uncond?, z_cond = True, z_dim = 1024, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed1-epoch=163-loss=0.0057 (copy 1).ckpt"
ddpm_path_list=($ddpm_path_1)

# loop over seeds
for seed in 1; do
  results_dir="${base_results_dir}cond_on_path/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
      --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
  done
done
