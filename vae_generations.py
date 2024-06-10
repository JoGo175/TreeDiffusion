
import os
import sys
import yaml
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from PIL import Image


# set directory to parent directory
os.chdir("/cluster/work/vogtlab/Group/jogoncalves/treevae")

# print current working directory
print("Current Working Directory:", os.getcwd())

from utils.data_utils import get_data, get_gen
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds, prepare_config, display_image
from models.model import TreeVAE


from utils.plotting_utils import plot_tree_graph

from utils.plotting_utils import plot_tree_graph



#mode = 'vae_recons'
#mode = 'vae_samples'


def vae_recons():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=f'{dataset}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--mode', default='vae_recons', type=str, help='evaluation mode: vae_recons or vae_samples')
    parser.add_argument('--model_name', default='', type=str, help='path to the pretrained TreeVAE model')
    
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mode = args.mode
    dataset = args.config_name
    ex_name = args.model_name

    path = '/Users/jorgegoncalves/Desktop/Repositories/Master_Thesis/treevae/models/experiments/'
    checkpoint_path = path+dataset+ex_name

    with open(checkpoint_path + "/config.yaml", 'r') as stream:
        configs = yaml.load(stream,Loader=yaml.Loader)
    print(configs)


    trainset, trainset_eval, testset = get_data(configs)
    gen_train = get_gen(trainset, configs, validation=False, shuffle=False)
    gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)

    y_train = trainset_eval.dataset.targets.clone().detach()[trainset_eval.indices].numpy()
    y_test = testset.dataset.targets.clone().detach()[testset.indices].numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TreeVAE(**configs['training'])
    data_tree = np.load(checkpoint_path+'/data_tree.npy', allow_pickle=True)

    model = construct_tree_fromnpy(model, data_tree, configs)
    if not (configs['globals']['eager_mode'] and configs['globals']['wandb_logging']!='offline'):
        #model = torch.compile(model)
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path+'/model_weights.pt', map_location=device), strict=True)
    model.to(device)
    model.eval()
        

    if mode == 'vae_recons':
        # get test set reconstructions

        # setup dirs
        vae_save_path = f"../results_all_leaves/{dataset}/seed_1/vae"
        img_save_path = os.path.join(vae_save_path, "recons_all_leaves")


        # loop over gen_test
        for j, (x, y) in enumerate(gen_test):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                res = model.compute_reconstruction(x)
                recons = res[0]
                nodes = res[1]

                num_leaves = len(nodes)

                # loop over each class and save every TreeVAE reconstruction of this class separately
                for c in range(num_leaves):
                    # Setup a dir for each class
                    class_save_pass = os.path.join(img_save_path, f"img_cluster_{c}")
                    os.makedirs(class_save_pass, exist_ok=True)
                    # save every image of this class separately
                    for i in range(x.shape[0]):
                        prob = nodes[c]['prob'][i].cpu()
                        fig, axs = plt.subplots(1, 1, figsize=(2, 2))
                        axs.imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                        axs.set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                        axs.axis('off')
                        # save image
                        plt.savefig(os.path.join(class_save_pass, f"output__{0}_{j}_{i}_{prob}.png"))
                        plt.close()


    elif mode == 'vae_samples':
        # get new generations

        # setup dirs
        vae_save_path = f"../results_all_leaves/{dataset}/seed_1/vae"
        img_save_path = os.path.join(vae_save_path, "sample_all_leaves")

        # loop over gen_test --> not really used, only to get again 10k
        for j, (x, y) in enumerate(gen_test):
            n_samples = x[0].size(0)
            reconstructions, p_c_z = model.generate_images(n_samples, x[0].device)

            num_leaves = len(reconstructions)

            # loop over each class and save every TreeVAE reconstruction of this class separately
            for c in range(num_leaves):
                # Setup a dir for each class
                class_save_pass = os.path.join(img_save_path, f"img_cluster_{c}")
                os.makedirs(class_save_pass, exist_ok=True)
                # save every image of this class separately
                for i in range(n_samples):
                    prob = p_c_z[c][i].cpu().numpy()
                    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
                    axs.imshow(display_image(reconstructions[c][i]), cmap=plt.get_cmap('gray'))
                    axs.set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs.axis('off')
                    # save image
                    plt.savefig(os.path.join(class_save_pass, f"output__{0}_{j}_{i}_{prob}.png"))
                    plt.close()

            

if __name__ == '__main__':
    vae_recons()


