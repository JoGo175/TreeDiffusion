"""
Given a trained TreeVAE model, this script generates reconstructions and samples for each leaf in the learned tree.
"""
import os
import yaml
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import get_data, get_gen
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import display_image
from models.model import TreeVAE

def vae_recons():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml', default='cifar10')
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    parser.add_argument('--mode', type=str, help='evaluation mode: vae_recons or vae_samples')
    parser.add_argument('--model_name', type=str, help='path to the pretrained TreeVAE model')
    
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mode = args.mode
    dataset = args.config_name
    ex_name = args.model_name

    path = 'models/experiments/'
    checkpoint_path = path+dataset+ex_name

    with open(checkpoint_path + "/config.yaml", 'r') as stream:
        configs = yaml.load(stream,Loader=yaml.Loader)
    print(configs)

    if dataset == "cubicc":
        configs['training']['batch_size'] = 32

    _, _, testset = get_data(configs)
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)

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
        
    # get test set reconstructions
    if mode == 'vae_recons':
        # setup dirs
        vae_save_path = f"../results_ICLR/{dataset}/cond_on_path/ddim/seed_1/vae"
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

    # get new generations
    elif mode == 'vae_samples':
        # setup dirs
        vae_save_path = f"../results_ICLR/{dataset}/cond_on_path/ddim/seed_1/vae"
        img_save_path = os.path.join(vae_save_path, "sample_all_leaves")

        # loop over gen_test --> not really used, only to get again 10k
        for j, (x, y) in enumerate(gen_test):
            n_samples = x.shape[0]
            reconstructions, p_c_z = model.generate_images(n_samples, device)
            num_leaves = len(reconstructions)

            # loop over each class and save every TreeVAE reconstruction of this class separately
            for c in range(num_leaves):
                # Setup a dir for each class
                class_save_pass = os.path.join(img_save_path, f"img_cluster_{c}")
                os.makedirs(class_save_pass, exist_ok=True)
                # save every image of this class separately
                for i in range(n_samples):
                    prob = p_c_z[i][c].cpu().detach()
                    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
                    axs.imshow(display_image(reconstructions[c][i].cpu().detach()), cmap=plt.get_cmap('gray'))
                    axs.set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs.axis('off')
                    # save image
                    plt.savefig(os.path.join(class_save_pass, f"output__{0}_{j}_{i}_{prob}.png"))
                    plt.close()


if __name__ == '__main__':
    vae_recons()