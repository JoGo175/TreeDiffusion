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
from utils.training_utils import compute_leaves, validate_one_epoch, Custom_Metrics, predict, move_to
import gc
from tqdm import tqdm

###############################################################################################################
# SELECT THE DATASET
dataset = "celeba"       # mnist, fmnist, cifar10, celeba, cubicc is supported
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
    gen_tain_eval = get_gen(trainset, configs_ddpm, validation=True, shuffle=False)
    gen_test = get_gen(testset, configs_ddpm, validation=True, shuffle=False)


    # setup device, mps is for M1 or M2 macs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "mps"

    base_path = '/cluster/work/vogtlab/Group/jogoncalves/treevae/'

    path_1 = 'models/experiments/celeba/20240918-032531_103db'
    path_2 = 'models/experiments/celeba/20240918-032554_0d008'
    path_3 = 'models/experiments/celeba/20240918-032615_143b9'
    path_4 = 'models/experiments/celeba/20240918-032635_0206c'
    path_5 = 'models/experiments/celeba/20240918-032751_46c3e'
    path_6 = 'models/experiments/celeba/20240918-032749_c73e9'
    path_7 = 'models/experiments/celeba/20240918-032937_80cf8'
    path_8 = 'models/experiments/celeba/20240918-032915_b4bde'
    path_9 = 'models/experiments/celeba/20240918-033153_8f6e7'
    path_10 = 'models/experiments/celeba/20240918-033153_3d955'
    vae_path_list = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10]

    # lists to store FID scores for each seed
    train_FID_generations_vae = []
    test_FID_generations_vae = []
    train_FID_reconstructions_vae = []
    test_FID_reconstructions_vae= []

    # load model
    for m, model_folder in enumerate(vae_path_list):
        checkpoint_path = base_path + model_folder

        model = TreeVAE(**configs['training'])
        data_tree = np.load(checkpoint_path + '/data_tree.npy', allow_pickle=True)
        model = construct_tree_fromnpy(model, data_tree, configs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(checkpoint_path + '/model_weights.pt', map_location=device), strict=False)
        model.to(device)
        model.eval()

        data_stats_train = get_precomputed_fid_scores_path("placeholder", configs['data']['data_name'],
                                                           batch_size=50, subset="train", device=device)
        data_stats_test = get_precomputed_fid_scores_path("placeholder", configs['data']['data_name'],
                                                          batch_size=50, subset="test", device=device)

        # Generations FID -----------------------------------------------------------

        # generate 10k samples from the model
        torch.cuda.empty_cache()
        n_imgs = 10000
        with torch.no_grad():
            generations, p_c_z = model.generate_images(n_imgs, device)
        generations = move_to(generations, 'cpu')
        p_c_z = move_to(p_c_z, 'cpu')

        # for each generated image, only save the ones that are in the leaf with the highest probability
        generations_list = []
        for i in range(n_imgs):
            # only save generation from leaf with highest probability
            leaf_ind = torch.argmax(p_c_z[i])
            generations_list.append(generations[leaf_ind][i])
        gen_dataset = torch.stack(generations_list).squeeze()
        _ = gc.collect()

        # compute FID score for generated images

        # precompute FID scores for generated images
        stats_generations = save_fid_stats_as_dict(gen_dataset, batch_size=50, device=device, dims=2048)
        train_FID_generations = calculate_fid([data_stats_train, stats_generations], batch_size=50, device=device,
                                              dims=2048)
        test_FID_generations = calculate_fid([data_stats_test, stats_generations], batch_size=50, device=device,
                                             dims=2048)

        print("FID score for generated images compared to train set:", train_FID_generations)
        print("FID score for generated images compared to test set:", test_FID_generations)
        train_FID_generations_vae.append(train_FID_generations)
        test_FID_generations_vae.append(test_FID_generations)
        _ = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reconstructions FID -----------------------------------------------------------

        # only compute reconstruction FID for test set in colored images to reduce cuda memory usage
        if configs['data']['data_name'] in ["cifar10", "cubicc", "celeba"]:
            fid_eval_set = ['test']
        else:
            fid_eval_set = ['train', 'test']

        for subset in fid_eval_set:
            reconstructions_list = []

            if subset == 'train':
                gen_train_eval = get_gen(trainset, configs, validation=True, shuffle=False)
                data_loader = gen_train_eval
            elif subset == 'test':
                gen_test = get_gen(testset, configs, validation=True, shuffle=False)
                data_loader = gen_test

            for inputs, labels in tqdm(data_loader):
                inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    reconstructions, node_leaves = model.compute_reconstruction(inputs_gpu)
                reconstructions = move_to(reconstructions, 'cpu')
                node_leaves = move_to(node_leaves, 'cpu')
                _ = gc.collect()

                # add reconstruction to list
                for i in range(len(inputs)):
                    # probs are the probabilities of each leaf for the data point i
                    probs = [node_leaves[j]['prob'][i] for j in range(len(node_leaves))]
                    # use the leaf with highest probability
                    leaf_ind = torch.argmax(torch.tensor(probs))
                    # add reconstruction to list
                    reconstructions_list.append(reconstructions[leaf_ind][i])
            reconstructions_dataset = torch.stack(reconstructions_list).squeeze().detach()
            _ = gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if configs['data']['data_name'] in ["cifar10", "cubicc", "celeba"]:
                batch_size_fid = 25
            else:
                batch_size_fid = 50

            # precompute FID scores for generated images
            stats_reconstructions = save_fid_stats_as_dict(reconstructions_dataset, batch_size=batch_size_fid,
                                                           device=device,
                                                           dims=2048)
            _ = gc.collect()

            if subset == 'train':
                train_FID_reconstructions = calculate_fid([data_stats_train, stats_reconstructions],
                                                          batch_size=batch_size_fid,
                                                          device=device, dims=2048)
                print("FID score for reconstructed images, train set:", train_FID_reconstructions)
                train_FID_reconstructions_vae.append(train_FID_reconstructions)
            elif subset == 'test':
                test_FID_reconstructions = calculate_fid([data_stats_test, stats_reconstructions],
                                                         batch_size=batch_size_fid,
                                                         device=device, dims=2048)
                print("FID score for reconstructed images, test set:", test_FID_reconstructions)
                test_FID_reconstructions_vae.append(test_FID_reconstructions)
            _ = gc.collect()

    print("\nFID scores for DDPM samples compared to train set:\n", train_FID_generations_vae)
    print("\nFID scores for DDPM samples compared to test set:\n", test_FID_generations_vae)
    print("\nFID scores for DDPM reconstructions compared to train set:\n", train_FID_reconstructions_vae)
    print("\nFID scores for DDPM reconstructions compared to test set:\n", test_FID_reconstructions_vae)

    # compute average and standard deviation of FID scores for DDPM
    print(configs['data']['data_name'])
    print("-----------------------------------")
    print("Reconstructions")
    print("\nAverage FID score for DDPM reconstructions compared to test set:",
          np.mean(test_FID_reconstructions_vae))
    print("\nStandard deviation FID score for DDPM reconstructions compared to test set:",
          np.std(test_FID_reconstructions_vae))
    print("-----------------------------------")
    print("Generations")
    print("\nAverage FID score for DDPM samples compared to test set:", np.mean(test_FID_generations_vae))
    print("\nStandard deviation FID score for DDPM samples compared to test set:",
          np.std(test_FID_generations_vae))




    #
    # if configs['data']['data_name'] in ['cubicc', 'celeba']:
    #     train_data = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
    #                     torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)])
    #     test_data = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
    #                                  torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)])
    # else:
    #     train_data = trainset.dataset.data
    #     test_data = testset.dataset.data
    # # precompute or load fid scores for train and test
    # data_stats_train = get_precomputed_fid_scores_path(train_data, configs['data']['data_name'],
    #                                                    subset="train", device=device)
    # data_stats_test = get_precomputed_fid_scores_path(test_data, configs['data']['data_name'], subset="test",
    #                                                   device=device)
    #
    # print("FID scores for train set:", data_stats_train)
    # print("FID scores for test set:", data_stats_test)
    #
    # # paths to the generated images
    # base_path = '../results_ICLR/'
    # exp_path = 'cond_on_path'
    # seed_list = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5", "seed_6", "seed_7", "seed_8", "seed_9", "seed_10"]
    #
    # # lists to store FID scores for each seed
    # cubicc_train_FID_generations_ddpm = []
    # cubicc_test_FID_generations_ddpm = []
    # cubicc_train_FID_reconstructions_ddpm = []
    # cubicc_test_FID_reconstructions_ddpm = []
    #
    # for i, seed in enumerate(seed_list):
    #     ddpm_samples_path = base_path + dataset + exp_path + "/ddim/" + seed + '/ddpm/sample/'
    #     ddpm_reconstructions_path = base_path + dataset + exp_path + "/ddim/" + seed + '/ddpm/recons/'
    #
    #     # load all images from the path and create a numpy array
    #     ddpm_samples = load_images_from_path(ddpm_samples_path)
    #     ddpm_reconstructions = load_images_from_path(ddpm_reconstructions_path)
    #
    #     # precompute FID scores for generated and reconstructed images of DDPM
    #
    #     #  DDPM samples
    #     ddpm_stats_generations = save_fid_stats_as_dict(ddpm_samples, batch_size=32, device=device, dims=2048)
    #     train_FID_gen_ddpm = calculate_fid([data_stats_train, ddpm_stats_generations], batch_size=32, device=device,
    #                                        dims=2048)
    #     test_FID_gen_ddpm = calculate_fid([data_stats_test, ddpm_stats_generations], batch_size=32, device=device,
    #                                       dims=2048)
    #     cubicc_train_FID_generations_ddpm.append(train_FID_gen_ddpm)
    #     cubicc_test_FID_generations_ddpm.append(test_FID_gen_ddpm)
    #
    #     #  DDPM reconstructions
    #     ddpm_stats_reconstructions = save_fid_stats_as_dict(ddpm_reconstructions, batch_size=32, device=device,
    #                                                         dims=2048)
    #     train_FID_recon_ddpm = calculate_fid([data_stats_train, ddpm_stats_reconstructions], batch_size=32,
    #                                          device=device, dims=2048)
    #     test_FID_recon_ddpm = calculate_fid([data_stats_test, ddpm_stats_reconstructions], batch_size=32, device=device,
    #                                         dims=2048)
    #     cubicc_train_FID_reconstructions_ddpm.append(train_FID_recon_ddpm)
    #     cubicc_test_FID_reconstructions_ddpm.append(test_FID_recon_ddpm)
    #
    # print("\nFID scores for DDPM samples compared to train set:\n", cubicc_train_FID_generations_ddpm)
    # print("\nFID scores for DDPM samples compared to test set:\n", cubicc_test_FID_generations_ddpm)
    # print("\nFID scores for DDPM reconstructions compared to train set:\n", cubicc_train_FID_reconstructions_ddpm)
    # print("\nFID scores for DDPM reconstructions compared to test set:\n", cubicc_test_FID_reconstructions_ddpm)
    #
    # # compute average and standard deviation of FID scores for DDPM
    # print("CUBICC")
    # print("-----------------------------------")
    # print("Reconstructions")
    # print("\nAverage FID score for DDPM reconstructions compared to test set:",
    #       np.mean(cubicc_test_FID_reconstructions_ddpm))
    # print("\nStandard deviation FID score for DDPM reconstructions compared to test set:",
    #       np.std(cubicc_test_FID_reconstructions_ddpm))
    # print("-----------------------------------")
    # print("Generations")
    # print("\nAverage FID score for DDPM samples compared to test set:", np.mean(cubicc_test_FID_generations_ddpm))
    # print("\nStandard deviation FID score for DDPM samples compared to test set:",
    #       np.std(cubicc_test_FID_generations_ddpm))

if __name__ == "__main__":
    train()