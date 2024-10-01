import copy
import numpy as np
import torch
import argparse
import distutils
import pytorch_lightning as pl
from pathlib import Path

from models.diffusion.callbacks import ImageWriter
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import SuperResModel, UNetModel
from models.model import TreeVAE
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds, prepare_config
from FID.fid_score import calculate_fid, get_precomputed_fid_scores_path, save_fid_stats_as_dict

###############################################################################################################
# SELECT THE DATASET
dataset = "cubicc"       # mnist, fmnist, cifar10, celeba, cubicc is supported
###############################################################################################################


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


def train():
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    # Get config and setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=f'{dataset}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--vae_chkpt_path', type=str, help='path to the pretrained TreeVAE model')
    parser.add_argument('--results_dir', type=str, help='path to the results directory')
    parser.add_argument('--chkpt_path', type=str, help='path to the pretrained DDPM model')
    parser.add_argument('--save_path', type=str, help='path to save the results')
    parser.add_argument('--eval_mode', type=str, help='evaluation mode: sample or recons')

    # conditioning arguments
    parser.add_argument('--ddpm_type', type=str, help='type of DDPM to train')
    parser.add_argument('--z_cond', type=lambda x: bool(distutils.util.strtobool(x)), help='use z as conditioning')
    parser.add_argument('--z_dim', type=int, help='dimension of latent space')
    parser.add_argument('--z_signal', type=str, help='type of z signal')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    # Configs specific to DDPM
    configs_ddpm = configs['ddpm']
    if args.seed is not None:
        configs_ddpm['globals']['seed'] = args.seed
    if args.eval_mode is not None:
        configs_ddpm['evaluation']['eval_mode'] = args.eval_mode
    if args.vae_chkpt_path is not None:
        configs_ddpm['training']['vae_chkpt_path'] = args.vae_chkpt_path
    if args.results_dir is not None:
        configs_ddpm['training']['results_dir'] = args.results_dir
    if args.chkpt_path is not None:
        configs_ddpm['evaluation']['chkpt_path'] = args.chkpt_path
    if args.save_path is not None:
        configs_ddpm['evaluation']['save_path'] = args.save_path
    if args.ddpm_type is not None:
        configs_ddpm['training']['type'] = args.ddpm_type
    if args.z_cond is not None:
        configs_ddpm['training']['z_cond'] = args.z_cond
    if args.z_dim is not None:
        configs_ddpm['training']['z_dim'] = args.z_dim
    if args.z_signal is not None:
        configs_ddpm['training']['z_signal'] = args.z_signal

    # Reproducibility
    reset_random_seeds(configs_ddpm['globals']['seed'])

    # Dataset
    trainset, trainset_eval, testset = get_data(configs_ddpm)
    gen_test = get_gen(testset, configs_ddpm, validation=True, shuffle=False)

    # setup device, mps is for M1 or M2 macs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if configs['data']['data_name'] in ['cubicc', 'celeba']:
        train_data = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
                        torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)])
        test_data = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
                                     torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)])
    else:
        train_data = trainset.dataset.data
        test_data = testset.dataset.data
    # precompute or load fid scores for train and test
    data_stats_train = get_precomputed_fid_scores_path(train_data, configs['data']['data_name'],
                                                       subset="train", device=device)
    data_stats_test = get_precomputed_fid_scores_path(test_data, configs['data']['data_name'], subset="test",
                                                      device=device)

    # paths to the generated images
    base_path = '../results_ICLR/'
    exp_path = 'cond_on_path'
    seed_list = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5", "seed_6", "seed_7", "seed_8", "seed_9", "seed_10"]

    # lists to store FID scores for each seed
    cubicc_train_FID_generations_ddpm = []
    cubicc_test_FID_generations_ddpm = []
    cubicc_train_FID_reconstructions_ddpm = []
    cubicc_test_FID_reconstructions_ddpm = []

    for i, seed in enumerate(seed_list):
        ddpm_samples_path = base_path + dataset + exp_path + "/ddim/" + seed + '/ddpm/sample/'
        ddpm_reconstructions_path = base_path + dataset + exp_path + "/ddim/" + seed + '/ddpm/recons/'

        # load all images from the path and create a numpy array
        ddpm_samples = load_images_from_path(ddpm_samples_path)
        ddpm_reconstructions = load_images_from_path(ddpm_reconstructions_path)

        # precompute FID scores for generated and reconstructed images of DDPM

        #  DDPM samples
        ddpm_stats_generations = save_fid_stats_as_dict(ddpm_samples, batch_size=32, device=device, dims=2048)
        train_FID_gen_ddpm = calculate_fid([data_stats_train, ddpm_stats_generations], batch_size=32, device=device,
                                           dims=2048)
        test_FID_gen_ddpm = calculate_fid([data_stats_test, ddpm_stats_generations], batch_size=32, device=device,
                                          dims=2048)
        cubicc_train_FID_generations_ddpm.append(train_FID_gen_ddpm)
        cubicc_test_FID_generations_ddpm.append(test_FID_gen_ddpm)

        #  DDPM reconstructions
        ddpm_stats_reconstructions = save_fid_stats_as_dict(ddpm_reconstructions, batch_size=32, device=device,
                                                            dims=2048)
        train_FID_recon_ddpm = calculate_fid([data_stats_train, ddpm_stats_reconstructions], batch_size=32,
                                             device=device, dims=2048)
        test_FID_recon_ddpm = calculate_fid([data_stats_test, ddpm_stats_reconstructions], batch_size=32, device=device,
                                            dims=2048)
        cubicc_train_FID_reconstructions_ddpm.append(train_FID_recon_ddpm)
        cubicc_test_FID_reconstructions_ddpm.append(test_FID_recon_ddpm)

    print("\nFID scores for DDPM samples compared to train set:\n", cubicc_train_FID_generations_ddpm)
    print("\nFID scores for DDPM samples compared to test set:\n", cubicc_test_FID_generations_ddpm)
    print("\nFID scores for DDPM reconstructions compared to train set:\n", cubicc_train_FID_reconstructions_ddpm)
    print("\nFID scores for DDPM reconstructions compared to test set:\n", cubicc_test_FID_reconstructions_ddpm)

    # compute average and standard deviation of FID scores for DDPM
    print("CUBICC")
    print("-----------------------------------")
    print("Reconstructions")
    print("\nAverage FID score for DDPM reconstructions compared to test set:",
          np.mean(cubicc_test_FID_reconstructions_ddpm))
    print("\nStandard deviation FID score for DDPM reconstructions compared to test set:",
          np.std(cubicc_test_FID_reconstructions_ddpm))
    print("-----------------------------------")
    print("Generations")
    print("\nAverage FID score for DDPM samples compared to test set:", np.mean(cubicc_test_FID_generations_ddpm))
    print("\nStandard deviation FID score for DDPM samples compared to test set:",
          np.std(cubicc_test_FID_generations_ddpm))

if __name__ == "__main__":
    train()