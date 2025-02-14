"""
This file has been modified from a file in the original DiffuseVAE reporitory
which was released under the MIT License, to adapt and improve it for the TreeVAE project.

Source:
https://github.com/kpandey008/DiffuseVAE?tab=readme-ov-file

---------------------------------------------------------------
MIT License

Copyright (c) 2021 Kushagra Pandey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------
"""
import logging
import os
import yaml
import argparse
import torch
import random

import pytorch_lightning as pl
import torchvision.transforms as T
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path

from vanilla_vae.vae import VAE
from utils.utils import reset_random_seeds
from utils.data_utils import get_data, get_gen

###############################################################################################################
# SELECT THE DATASET
dataset_name = "mnist"       # mnist, fmnist, cifar10, celeba, cubicc is supported
###############################################################################################################


def load_config(args, project_dir):
	# Load config
	data_name = args.config_name +'.yml'
	config_path = project_dir / 'vanilla_vae/configs' / data_name

	with config_path.open(mode='r') as yamlfile:
		configs = yaml.safe_load(yamlfile)
	
	return configs



def train():
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    # Get config and setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=f'{dataset_name}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', type=int, help='random seed')

    args = parser.parse_args()
    config = load_config(args, project_dir)

    # Get config and setup
    config = config["vae"]
    if args.seed is not None:
         config['globals']['seed'] = args.seed

    # Reproducibility
    reset_random_seeds(config['globals']['seed'])

    # Dataset
    trainset, trainset_eval, testset = get_data(config)
    gen_train = get_gen(trainset, config, validation=False, shuffle=False)

    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]

    # Model
    vae = VAE(
        input_res=image_size,
        enc_block_str=config["model"]["enc_block_config"],
        dec_block_str=config["model"]["dec_block_config"],
        enc_channel_str=config["model"]["enc_channel_config"],
        dec_channel_str=config["model"]["dec_channel_config"],
        lr=config["training"]["lr"],
        alpha=config["training"]["alpha"],
    )

    # Trainer
    train_kwargs = {}
    restore_path = config["training"]["restore_path"]
    if restore_path is not None:
        # Restore checkpoint
        pass
        # train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config["training"]["results_dir"]
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config['training']['chkpt_prefix']}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config["training"]["chkpt_interval"],
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config["training"]["epochs"]
    train_kwargs["log_every_n_steps"] = config["training"]["log_step"]
    train_kwargs["callbacks"] = [chkpt_callback]

    # # device
    # if torch.cuda.is_available():
    #     train_kwargs["gpus"] = 1
    #     train_kwargs["accelerator"] = "gpu"
    # else:
    #     train_kwargs["gpus"] = None
    #     train_kwargs["accelerator"] = None


    # Half precision training
    if config["training"]["fp16"]:
        train_kwargs["precision"] = 16


    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloaders=gen_train)


if __name__ == "__main__":
    train()