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
import copy
import yaml
import os
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
import distutils
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from models.diffusion.callbacks import EMAWeightUpdate
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import UNetModel, SuperResModel
from vanilla_vae.vae import VAE
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds

###############################################################################################################
# SELECT THE DATASET
dataset = "mnist"       # mnist, fmnist, cifar10, celeba, cubicc is supported
###############################################################################################################


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


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
    parser.add_argument('--config_name', default=f'{dataset}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--vae_chkpt_path', type=str, help='path to the pretrained TreeVAE model')
    parser.add_argument('--results_dir', type=str, help='path to the results directory')
    parser.add_argument('--chkpt_prefix', type=str, help='prefix for the checkpoint file')
    
    # conditioning arguments
    parser.add_argument('--ddpm_type', type=str, help='type of DDPM to train')

    args = parser.parse_args()
    configs = load_config(args, project_dir)

    # Configs specific to DDPM
    configs_ddpm = configs['ddpm']
    if args.seed is not None:
        configs_ddpm['globals']['seed'] = args.seed
    if args.vae_chkpt_path is not None:
        configs_ddpm['training']['vae_chkpt_path'] = args.vae_chkpt_path
    if args.results_dir is not None:
        configs_ddpm['training']['results_dir'] = args.results_dir
    if args.chkpt_prefix is not None:
        configs_ddpm['training']['chkpt_prefix'] = args.chkpt_prefix
    if args.ddpm_type is not None:
        configs_ddpm['training']['type'] = args.ddpm_type

    # Reproducibility
    reset_random_seeds(configs_ddpm['globals']['seed'])

    # Dataset
    trainset, _, _ = get_data(configs_ddpm)
    gen_train = get_gen(trainset, configs_ddpm, validation=False, shuffle=False)

    # UNet Denoising Model for DDPM
    attn_resolutions = __parse_str(configs_ddpm["model"]["attn_resolutions"])
    dim_mults = __parse_str(configs_ddpm["model"]["dim_mults"])
    ddpm_type = configs_ddpm["training"]["type"]
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=configs_ddpm["data"]["inp_channels"],
        model_channels=configs_ddpm["model"]["dim"],
        out_channels=configs_ddpm["data"]["inp_channels"],
        num_res_blocks=configs_ddpm["model"]["n_residual"],
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=configs_ddpm["model"]["dropout"],
        num_heads=configs_ddpm["model"]["n_heads"],
        z_dim=configs_ddpm["training"]["z_dim"],
        use_scale_shift_norm=configs_ddpm["training"]["z_cond"],
        use_z=configs_ddpm["training"]["z_cond"],
    )

    # EMA (Exponential Moving Average) parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    # DDPM framework for conditional training, aka refiner
    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=configs_ddpm["model"]["beta1"],
        beta_2=configs_ddpm["model"]["beta2"],
        T=configs_ddpm["model"]["n_timesteps"],
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=configs_ddpm["model"]["beta1"],
        beta_2=configs_ddpm["model"]["beta2"],
        T=configs_ddpm["model"]["n_timesteps"],
    )

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)

    # Load pretrained TreeVAE model, aka generator
    model_path = configs_ddpm["training"]["vae_chkpt_path"]
    vae = VAE.load_from_checkpoint(
        configs_ddpm["training"]["vae_chkpt_path"],
        input_res=configs_ddpm["data"]["image_size"],
    )
    vae.eval()

    # Freeze all parameters of VAE as they are non-trainable
    for p in vae.parameters():
        p.requires_grad = False

    # Wrapper function for the whole Diffuse-TreeVAE model
    ddpm_wrapper = DDPMWrapper(
        online_ddpm,
        target_ddpm,
        vae,
        lr=configs_ddpm["training"]["lr"],
        cfd_rate=configs_ddpm["training"]["cfd_rate"],
        n_anneal_steps=configs_ddpm["training"]["n_anneal_steps"],
        loss=configs_ddpm["training"]["loss"],
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=configs_ddpm["training"]["grad_clip"],
        z_cond=configs_ddpm["training"]["z_cond"],
    )

    # Trainer settings
    train_kwargs = {}
    restore_path = configs_ddpm["training"]["restore_path"]
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = configs_ddpm["training"]["results_dir"]
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{configs_ddpm['training']['chkpt_prefix']}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=configs_ddpm["training"]["chkpt_interval"],
        save_on_train_epoch_end=True,
    )

    # Training parameters
    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = configs_ddpm["training"]["epochs"]
    train_kwargs["callbacks"] = [chkpt_callback]

    if configs_ddpm["training"]["use_ema"]:
        ema_callback = EMAWeightUpdate(tau=configs_ddpm["training"]["ema_decay"])
        train_kwargs["callbacks"].append(ema_callback)

    # Start training
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloaders=gen_train)


if __name__ == "__main__":
    train()