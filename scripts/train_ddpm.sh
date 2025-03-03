#!/bin/bash
#SBATCH --time=168:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae


# Base directories
base_model_dir="vanilla_vae/results"
base_results_dir="/cluster/work/vogtlab/Group/jogoncalves/Repos/TreeDiffusion/vanilla_vae/results"
log_base="/cluster/work/vogtlab/Group/jogoncalves/logs"

# List of datasets to process
datasets=("celeba")  # "mnist" "fmnist" "cifar10" "celeba" "cubicc"

# Define checkpoint names for each dataset.
checkpoints_mnist=(
  'vae-vae_1-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_2-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_3-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_4-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_5-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_6-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_7-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_8-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_9-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_10-epoch=1499-train_loss=0.0000.ckpt'
)
checkpoints_fmnist=(
  'vae-vae_1-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_2-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_3-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_4-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_5-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_6-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_7-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_8-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_9-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_10-epoch=1499-train_loss=0.0000.ckpt'
)
checkpoints_cifar10=(
  'vae-vae_1-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_2-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_3-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_4-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_5-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_6-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_7-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_8-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_9-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_10-epoch=1499-train_loss=0.0000.ckpt'
)
checkpoints_celeba=(
  'vae-vae_1-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_2-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_3-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_4-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_5-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_6-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_7-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_8-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_9-epoch=349-train_loss=0.0000.ckpt'
  'vae-vae_10-epoch=349-train_loss=0.0000.ckpt'
)
checkpoints_cubicc=(
  'vae-vae_1-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_2-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_3-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_4-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_5-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_6-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_7-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_8-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_9-epoch=1499-train_loss=0.0000.ckpt'
  'vae-vae_10-epoch=1499-train_loss=0.0000.ckpt'
)

# Loop over each dataset
for dataset in "${datasets[@]}"; do
  # Get the corresponding checkpoint array via indirect expansion
  checkpoint_array_name="checkpoints_${dataset}"

# Set the GPU resource flag based on dataset name
  if [[ "$dataset" == "celeba" || "$dataset" == "cubicc" ]]; then
    gres_flag="--gres=gpu:rtx4090:1"
  else
    gres_flag="--gres=gpu:1"
  fi
  
  # Define the results directory and log output path
  results_dir="${base_results_dir}/${dataset}/"
  O_DIR="${log_base}/output.%x.%J_${dataset}.out"
  
  # Loop over seeds 1 to 10
  for seed in 6; do # {1..10}; do
    chkpt_prefix="ddpm_seed${seed}"
    # Array indices start at 0
    eval "checkpoint=\${${checkpoint_array_name}[$((seed - 1))]}"
    full_checkpoint_path="${base_model_dir}/${dataset}/checkpoints/${checkpoint}"
    
    echo "Submitting job for dataset: $dataset, seed: $seed, checkpoint: $full_checkpoint_path"
    sbatch --time=168:00:00 \
           --mem-per-cpu=20G \
           -p gpu \
            $gres_flag \
           -A vogtlab \
           --tmp=20G \
           --cpus-per-task=2 \
           -o "$O_DIR" \
           --wrap="python train_ddpm.py --config_name $dataset --chkpt_prefix $chkpt_prefix --vae_chkpt_path $full_checkpoint_path --results_dir $results_dir --seed $seed"
  done
done