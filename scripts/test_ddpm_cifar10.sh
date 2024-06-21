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

# vae_chkpt_path strings to the trained TreeVAE models
vae_chkpt_path='/cluster/work/vogtlab/Group/jogoncalves/treevae/models/experiments/cifar10/20240613-111829_7bc09'


# Fully unconditional
# type = “uncond”, z_cond = False, z_dim = None, z_signal = None

ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/fully_uncond/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0163.ckpt'
ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/fully_uncond/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0151.ckpt'
ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/fully_uncond/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0165.ckpt'
ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)

# loop over seeds and vae_chkpt_path
for seed in 1 2 3; do
  results_dir="${base_results_dir}fully_uncond/seed_${seed}/"
  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
  for eval_mode in 'sample' 'recons'; do
    # run the job
    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond False"
  done
done

#
#
## Conditioning on Leaf Reconstructions
## type = “form1”, z_cond = False, z_dim = None, z_signal = None
#
#ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0151.ckpt'
#ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0139.ckpt'
#ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0152.ckpt'
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)
#
## loop over seeds and vae_chkpt_path
#for seed in 1 2 3; do
#  results_dir="${base_results_dir}cond_on_recons/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'sample_all_leaves' 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond False"
#  done
#done
#

#
## Conditioning on Leaf Reconstructions + Leaf Index
## type = “form1”, z_cond = True, z_dim = 1, z_signal = “cluster_id”
#
#ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0151.ckpt'
#ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0169.ckpt'
#ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0201.ckpt'
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)
#
## loop over seeds and vae_chkpt_path
#for seed in 1 2 3; do
#  results_dir="${base_results_dir}cond_on_recons_and_index/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'sample_all_leaves' 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1 --z_signal cluster_id"
#  done
#done
#
#
#
## Conditioning on Leaf Reconstructions + Leaf Embeddings
## type = “form1”, z_cond = True, z_dim = 1024, z_signal = “latent”
#
#ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_emb/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0136.ckpt'
#ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_emb/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0137.ckpt'
#ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_emb/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0200.ckpt'
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)
#
## loop over seeds and vae_chkpt_path
#for seed in 1 2 3; do
#  results_dir="${base_results_dir}cond_on_recons_and_emb/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'sample_all_leaves' 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal latent"
#  done
#done
#
#
#
#
## Conditioning on Leaf Reconstructions + Leaf Index + Leaf Embeddings
## type = “form1”, z_cond = True, z_dim = 1024, z_signal = “both”
#
#ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index_and_emb/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0136.ckpt'
#ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index_and_emb/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0136.ckpt'
#ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_recons_and_index_and_emb/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0151.ckpt'
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)
#
## loop over seeds and vae_chkpt_path
#for seed in 1 2 3; do
#  results_dir="${base_results_dir}cond_on_recons_and_index_and_emb/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'sample_all_leaves' 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'form1' --z_cond True --z_dim 1024 --z_signal both"
#  done
#done
#
#
## Conditioning on Leaf Index + Leaf Embeddings
## type = “uncond”, z_cond = True, z_dim = 1024, z_signal = “both”
#
#ddpm_path_1='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_index_and_emb/seed_1/checkpoints/ddpmv2-vae-epoch=999-loss=0.0152.ckpt'
#ddpm_path_2='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_index_and_emb/seed_2/checkpoints/ddpmv2-vae-epoch=999-loss=0.0139.ckpt'
#ddpm_path_3='/cluster/work/vogtlab/Group/jogoncalves/results_latent_emb/cifar10/cond_on_index_and_emb/seed_3/checkpoints/ddpmv2-vae-epoch=999-loss=0.0202.ckpt'
#ddpm_path_list=($ddpm_path_1 $ddpm_path_2 $ddpm_path_3)
#
## loop over seeds and vae_chkpt_path
#for seed in 1 2 3; do
#  results_dir="${base_results_dir}cond_on_index_and_emb/seed_${seed}/"
#  # loop over eval_mode = ['sample', 'sample_all_leaves', 'recons', 'recons_all_leaves']
#  for eval_mode in 'sample_all_leaves' 'recons_all_leaves'; do
#    # run the job
#    sbatch --time=100:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=1 -o $O_DIR --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $vae_chkpt_path --chkpt_path ${ddpm_path_list[$seed-1]} --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode --ddpm_type 'uncond' --z_cond True --z_dim 1024 --z_signal both"
#  done
#done
