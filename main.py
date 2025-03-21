"""
Runs the treeVAE model.
"""
import argparse
from pathlib import Path
import distutils

from train.train import run_experiment
from utils.utils import prepare_config

def main():
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--data_name', type=str, help='the dataset')
    parser.add_argument('--num_epochs', type=int, help='the number of training epochs')
    parser.add_argument('--num_epochs_finetuning', type=int, help='the number of finetuning epochs')
    parser.add_argument('--intermediate_fulltrain', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to finetune the full model during growing')
    parser.add_argument('--num_epochs_intermediate_fulltrain', type=int, help='the number of finetuning epochs during training')
    parser.add_argument('--num_epochs_smalltree', type=int, help='the number of sub-tree training epochs')
    parser.add_argument('--num_clusters_data', type=int, help='the number of clusters in the data')
    parser.add_argument('--num_clusters_tree', type=int, help='the max number of leaves of the tree')
    parser.add_argument('--spectral_norm', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to use spectral normalization')

    # KL annealing parameters
    parser.add_argument('--kl_start', type=float, nargs='?', const=0., help='initial KL divergence from where annealing starts')
    parser.add_argument('--decay_kl', type=float, help='KL divergence annealing')

    # Model parameters
    parser.add_argument('--latent_channels', type=str, help='specifies the latent channels of the tree')
    parser.add_argument('--bottom_up_channels', type=str, help='specifies how many channels should the bottom-up model have')
    parser.add_argument('--representation_dim', type=int, help='the dimension of the latent representation')
    parser.add_argument('--dropout_router', type=float, help='dropout rate for the router')
    parser.add_argument('--res_connections', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to use residual connections in transformations and bottom-up model layers')
    parser.add_argument('--grow', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to grow the tree')
    parser.add_argument('--act_function', type=str, help='activation function')
    parser.add_argument('--dim_mod_conv', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to change the dimensionality via convolutional layers')

    # Contrastive learning parameters
    parser.add_argument('--augment', type=lambda x:bool(distutils.util.strtobool(x)), help='augment images or not')
    parser.add_argument('--augmentation_method', type=str, help='none vs simple augmentation vs contrastive approaches')
    parser.add_argument('--aug_decisions_weight', type=float, help='weight of similarity regularizer for augmented images')

    # Evaluation parameters
    parser.add_argument('--compute_ll', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to compute the log-likelihood')

    # Other parameters
    parser.add_argument('--save_model', type=lambda x: bool(distutils.util.strtobool(x)), help='specifies if the model should be saved')
    parser.add_argument('--eager_mode', type=lambda x: bool(distutils.util.strtobool(x)), help='specifies if the model should be run in graph or eager mode')
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader')
    parser.add_argument('--seed', type=int, help='random number generator seed')
    parser.add_argument('--wandb_logging', type=str, help='online, disabled, offline enables logging in wandb')

    # Specify config name
    parser.add_argument('--config_name', default='cubicc', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    run_experiment(configs)


if __name__ == "__main__":
    main()
