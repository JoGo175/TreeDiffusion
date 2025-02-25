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
datasets=("mnist" "fmnist" "cifar10" "cubicc")  # "mnist" "fmnist" "cifar10" "celeba" "cubicc"

# Define VAE checkpoint names for each dataset
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
  'model1'
  'model2'
  'model3'
  'model4'
  'model5'
  'model6'
  'model7'
  'model8'
  'model9'
  'model10'
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

# Define DDPM checkpoint names for each dataset
ddpm_checkpoints_mnist=(
  'model1'
  'model2'
  'model3'
  'model4'
  'model5'
  'model6'
  'model7'
  'model8'
  'model9'
  'model10'
)
ddpm_checkpoints_fmnist=(
  'model1'
  'model2'
  'model3'
  'model4'
  'model5'
  'model6'
  'model7'
  'model8'
  'model9'
  'model10'
)
ddpm_checkpoints_cifar10=(
  'model1'
  'model2'
  'model3'
  'model4'
  'model5'
  'model6'
  'model7'
  'model8'
  'model9'
  'model10'
)
ddpm_checkpoints_cubicc=(
  'model1'
  'model2'
  'model3'
  'model4'
  'model5'
  'model6'
  'model7'
  'model8'
  'model9'
  'model10'
)

# Loop over each dataset
for dataset in "${datasets[@]}"; do
  # Get the corresponding VAE checkpoint array via indirect expansion
  vae_array_name="checkpoints_${dataset}"
  
  # Get the corresponding DDPM checkpoint array via indirect expansion
  ddpm_array_name="ddpm_checkpoints_${dataset}"
  
  # Define the results directory and log output path
  results_dir="${base_results_dir}/${dataset}/"
  O_DIR="${log_base}/output.%x.%J_${dataset}.out"
  
  # Loop over seeds 1 to 10
  for seed in {1..10}; do
    # Array indices start at 0
    vae_checkpoint="${!vae_array_name[$((seed-1))]}"
    ddpm_checkpoint="${!ddpm_array_name[$((seed-1))]}"
    
    # Build full checkpoint paths
    full_vae_path="${base_model_dir}/${dataset}/checkpoints/${vae_checkpoint}"
    full_ddpm_path="${base_model_dir}/${dataset}/checkpoints/${ddpm_checkpoint}"
    
    # Loop over evaluation modes: "sample" and "recons"
    for eval_mode in "sample" "recons"; do
      echo "Submitting test job for dataset: $dataset, seed: $seed, eval_mode: $eval_mode"
      sbatch --time=100:00:00 \
             --mem-per-cpu=20G \
             -p gpu \
             --gres=gpu:1 \
             -A vogtlab \
             --tmp=20G \
             --cpus-per-task=1 \
             -o "$O_DIR" \
             --wrap="python test_ddpm.py --config_name $dataset --vae_chkpt_path $full_vae_path --chkpt_path $full_ddpm_path --results_dir $results_dir --save_path $results_dir --seed $seed --eval_mode $eval_mode"
    done
  done
done