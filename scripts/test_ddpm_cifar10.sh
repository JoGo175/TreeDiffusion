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
path_1='models/experiments/cifar10/20240906-175406_3bd31'
path_2='models/experiments/cifar10/20240906-175504_9d17f'
path_3='models/experiments/cifar10/20240906-175709_3858a'
path_4='models/experiments/cifar10/20240906-180213_73aee'
path_5='models/experiments/cifar10/20240906-182135_69bf1'
path_6='models/experiments/cifar10/20240906-183807_45e5f'
path_7='models/experiments/cifar10/20240906-184142_8a17a'
path_8='models/experiments/cifar10/20240906-190023_8637a'
path_9='models/experiments/cifar10/20240906-190441_5a2c6'
path_10='models/experiments/cifar10/20240906-190658_37148'
vae_path_list=($path_1 $path_2 $path_3 $path_4 $path_5 $path_6 $path_7 $path_8 $path_9 $path_10)

# directory to save the results
base_results_dir='/cluster/work/vogtlab/Group/jogoncalves/results_ICLR/cifar10/'


# Fully unconditional
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

ddpm_path_1="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0128"
ddpm_path_2="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0193"
ddpm_path_3="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0196"
ddpm_path_4="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0138"
ddpm_path_5="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0127"
ddpm_path_6="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0184"
ddpm_path_7="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0174"
ddpm_path_8="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0168"
ddpm_path_9="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0150"
ddpm_path_10="${base_results_dir}fully_uncond/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0137"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0119"
ddpm_path_2="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0138"
ddpm_path_3="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0121"
ddpm_path_4="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0140"
ddpm_path_5="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0174"
ddpm_path_6="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0150"
ddpm_path_7="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0110"
ddpm_path_8="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0123"
ddpm_path_9="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0116"
ddpm_path_10="${base_results_dir}cond_on_recons/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0132"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0117"
ddpm_path_2="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0136"
ddpm_path_3="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0121"
ddpm_path_4="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0079"
ddpm_path_5="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0172"
ddpm_path_6="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0137"
ddpm_path_7="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0118"
ddpm_path_8="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0165"
ddpm_path_9="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0090"
ddpm_path_10="${base_results_dir}cond_on_recons_and_index/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0081"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0117"
ddpm_path_2="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0136"
ddpm_path_3="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0121"
ddpm_path_4="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0079"
ddpm_path_5="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0172"
ddpm_path_6="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0137"
ddpm_path_7="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0118"
ddpm_path_8="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0165"
ddpm_path_9="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0090"
ddpm_path_10="${base_results_dir}cond_on_recons_and_emb/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0081"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0128"
ddpm_path_2="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0139"
ddpm_path_3="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0094"
ddpm_path_4="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0079"
ddpm_path_5="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0152"
ddpm_path_6="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0135"
ddpm_path_7="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0106"
ddpm_path_8="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0069"
ddpm_path_9="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0095"
ddpm_path_10="${base_results_dir}cond_on_recons_and_index_and_emb/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0103"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0118"
ddpm_path_2="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0140"
ddpm_path_3="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0139"
ddpm_path_4="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0079"
ddpm_path_5="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0195"
ddpm_path_6="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0138"
ddpm_path_7="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0108"
ddpm_path_8="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0165"
ddpm_path_9="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0095"
ddpm_path_10="${base_results_dir}cond_on_index_and_emb/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0132"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3 4 5 6 7 8 9 10; do
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

ddpm_path_1="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0144"
ddpm_path_2="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0140"
ddpm_path_3="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0140"
ddpm_path_4="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0082"
ddpm_path_5="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed5-epoch=999-loss=0.0198"
ddpm_path_6="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0139"
ddpm_path_7="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed7-epoch=986-loss=0.0133"
ddpm_path_8="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0166"
ddpm_path_9="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0097"
ddpm_path_10="${base_results_dir}cond_on_recons_and_path/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0132"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_recons_and_path/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal path"
done


# Conditioning on Path
# type = ?uncond?, z_cond = True, z_dim = 1024, z_signal = ?path?

ddpm_path_1="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed1-epoch=999-loss=0.0146"
ddpm_path_2="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed2-epoch=999-loss=0.0142"
ddpm_path_3="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed3-epoch=999-loss=0.0143"
ddpm_path_4="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed4-epoch=999-loss=0.0083"
ddpm_path_5="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed5-epoch=994-loss=0.0160"
ddpm_path_6="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed6-epoch=999-loss=0.0142"
ddpm_path_7="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed7-epoch=999-loss=0.0113"
ddpm_path_8="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed8-epoch=999-loss=0.0168"
ddpm_path_9="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed9-epoch=999-loss=0.0094"
ddpm_path_10="${base_results_dir}cond_on_path/checkpoints/ddpmv2-vae_seed10-epoch=999-loss=0.0135"
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3 $ddpm_path_4 $ddpm_path_5 $ddpm_path_6 $ddpm_path_7 $ddpm_path_8 $ddpm_path_9 $ddpm_path_10)

# loop over seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  results_dir="${base_results_dir}cond_on_path/seed_${seed}/"
  # run the job
  sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR \
    --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path ${vae_path_list[$seed-1]} --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal path"
done
