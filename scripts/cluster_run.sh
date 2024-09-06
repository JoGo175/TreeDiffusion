#!/bin/bash
#SBATCH --time=36:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="mnist"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"


# loop over kl_start
for kl_start in 0.0 0.5; do     # use 0.01 instead of 0.0 for CIFAR10!
  # loop over spectral_norm = False
  for res_connections in True False; do
    # loop over seeds
    for seed in 1 2 3 4 5 6 7 8 9 10; do
      # run the job
      sbatch --time=36:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --kl_start $kl_start --res_connections $res_connections --seed $seed"
    done
  done
done


# # loop over latent_channels in [64, 64, 64, 64, 64, 64, 64], [8, 16, 32, 64, 128, 256, 512], [512 256 128 64 32 16 8]
# for latent_channels in "64,64,64,64,64,64,64" "8,16,32,64,128,256,512" "512,256,128,64,32,16,8"; do
#   # loop over bottom_up_channels in [128, 128, 128, 128, 128, 128, 128], [16, 32, 64, 128, 256, 512, 1024], [1024, 512, 256, 128, 64, 32, 16]
#   for bottom_up_channels in "128,128,128,128,128,128,128" "16,32,64,128,256,512,1024" "1024,512,256,128,64,32,16"; do
#     # loop over seeds
#     for seed in 1 2 3 4 5 6 7 8 9 10; do
#       # run the job
#       sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --latent_channels $latent_channels --bottom_up_channels $bottom_up_channels --seed $seed"
#     done
#   done
# done




## loop over dim_mod_conv
#for dim_mod_conv in True False; do
#  # loop over representation_dim
#  for representation_dim in 1 2 4; do
#    # loop over seeds
#    for seed in 1 2 3 4 5 6 7 8 9 10; do
#      # run the job
#      sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=10G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --representation_dim $representation_dim --dim_mod_conv $dim_mod_conv --seed $seed"
#    done
#  done
#done
#


# # loop over kl_start
# for kl_start in 0.0 0.5; do     # use 0.01 instead of 0.0 for CIFAR10!
#   # loop over spectral_norm = False
#   for spectral_norm in True False; do
#     # loop over act_function in ["swish", "leaky_relu"]
#     for act_function in "swish" "leaky_relu"; do
#       # loop over res_connections in [True, False]
#       for res_connections in True False; do
#         # loop over seeds
#         for seed in 1 2 3 4 5 6 7 8 9 10; do
#           # run the job
#           sbatch --time=36:00:00 --mem-per-cpu=20G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --kl_start $kl_start --spectral_norm $spectral_norm --act_function $act_function --res_connections $res_connections --seed $seed"
#         done
#       done
#     done
#   done
# done



# for kl_start in 0.0 0.5 1; do
#     # loop over seeds 
#     for seed in 1 2 3 4 5 6 7 8 9 10; do
#         # run the job
#         sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --kl_start $kl_start --seed $seed"
#         # wait for 5 seconds
#         sleep 3
#     done
# done


# # loop over seeds
# for seed in 1 2 3 4 5 6 7 8 9 10; do
#   # run the job
#   sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=10G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model False --config_name $dataset --seed $seed"
#   # wait for 5 seconds
#   sleep 3
# done


# # loop over size of latent space [1, 2, 3, 4, 5, 6, 7 ,8]
# for depth in 1 2 3 4 5 6 7 8; do
#   # create the latent_channels and bottom_up_channels string with depth
#   latent_channels="10"
#   bottom_up_channels="16"
#   for ((i=1; i<$depth; i++)); do
#     latent_channels="${latent_channels},10"
#     bottom_up_channels="${bottom_up_channels},16"
#   done
#   # loop over seeds
#   for seed in 1 2 3 4 5 6 7 8 9 10; do
#     # run the job
#     sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=10G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model False --config_name $dataset --latent_channels $latent_channels --bottom_up_channels $bottom_up_channels --seed $seed"
#     # wait for 5 seconds
#     sleep 3
#   done
# done


### loop over latent_channels in [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]
#for latent_c in 32 64 128; do
# # create the latent_channels string [latent_c, latent_c, latent_c, latent_c, latent_c, latent_c]
# latent_channels="${latent_c},${latent_c},${latent_c},${latent_c},${latent_c},${latent_c}"
# for bottom_up_c in 16 32; do
#   bottom_up_channels="${bottom_up_c},${bottom_up_c},${bottom_up_c},${bottom_up_c},${bottom_up_c},${bottom_up_c}"
#   # loop over seeds
#   for seed in 1 2 3 4 5 6 7 8 9 10; do
#     # run the job
#     sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model False --config_name $dataset --latent_channels $latent_channels --bottom_up_channels $bottom_up_channels --seed $seed"
#     # wait for 5 seconds
#     sleep 3
#   done
# done
# # wait for 60min to avoid overloading the cluster
# sleep 3
#done



#
## loop over kl_start
#for representation_dim in 1 2 4 8 16; do
#  # loop over seeds
#  for seed in 1 2 3 4 5 6 7 8 9 10; do
#    # run the job
#    sbatch --time=36:00:00 --mem-per-cpu=10G -p gpu --gres=gpu:1 -A vogtlab --tmp=10G --cpus-per-task=2 -o $O_DIR --wrap="python main.py --save_model True --config_name $dataset --representation_dim $representation_dim --seed $seed"
#    # wait for 5 seconds
#    sleep 3
#  done
#done



