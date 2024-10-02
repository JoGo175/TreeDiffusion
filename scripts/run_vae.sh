#!/bin/bash
#SBATCH --time=36:00:00
eval "$(conda shell.bash hook)"
# Needed for deterministic computations
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
source ~/.bashrc
conda activate treevae
dataset="cubicc"
O_DIR="/cluster/work/vogtlab/Group/jogoncalves/logs/output.%x.%J_${dataset}.out"

if [ "$dataset" == "mnist" ]; then
  model_name="/20240906-142239_b81c6"
elif [ "$dataset" == "fmnist" ]; then
  model_name="/20240906-144022_3c565"
elif [ "$dataset" == "cifar10" ]; then
  model_name="/20240906-175406_3bd31"
elif [ "$dataset" == "cubicc" ]; then
  model_name="/20240923-013355_8ddcc"
fi


for mode in 'vae_samples'; do
  sbatch --time=36:00:00 --mem-per-cpu=15G -p gpu --gres=gpu:1 -A vogtlab --tmp=20G --cpus-per-task=2 -o $O_DIR --wrap="python vae_generations.py --config_name $dataset --mode $mode --model_name $model_name"
done
