import copy
import yaml
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
from vanilla_vae.vae import VAE
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds
import itertools
from torch.utils.data import Subset

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
    parser.add_argument('--chkpt_path', type=str, help='path to the pretrained DDPM model')
    parser.add_argument('--save_path', type=str, help='path to save the results')
    parser.add_argument('--eval_mode', type=str, help='evaluation mode: sample or recons')

    # conditioning arguments
    parser.add_argument('--ddpm_type', type=str, help='type of DDPM to train')

    args = parser.parse_args()
    configs = load_config(args, project_dir)
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

    # Reproducibility
    reset_random_seeds(configs_ddpm['globals']['seed'])

    # Dataset
    _, _, testset = get_data(configs_ddpm)


    # if we are sampling instead of reconstructing, we do not use the gen_test data itself,
    # only use the sequentiality of the dataloader, but need the appropriate sample length
    if configs_ddpm["evaluation"]["eval_mode"] in ["sample"]:
        # need to "stretch" the dataloader to size of n_samples
        n_samples = configs_ddpm["evaluation"]["n_samples"]
        current_len = len(testset)

        if current_len > n_samples:  # Truncate if too large
            testset = Subset(testset.dataset, testset.indices[:n_samples])

        elif current_len < n_samples:  # Stretch if too small
            extended_indices = list(itertools.islice(itertools.cycle(testset.indices), n_samples))
            testset = Subset(testset.dataset, extended_indices)

    gen_test = get_gen(testset, configs_ddpm, validation=True, shuffle=False)



    # Pre-sampled latents for DDPM if available
    ddpm_latent_path = configs_ddpm["data"]["ddpm_latent_path"]
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained TreeVAE model, aka generator
    model_path = configs_ddpm["training"]["vae_chkpt_path"]
    vae = VAE.load_from_checkpoint(
        configs_ddpm["training"]["vae_chkpt_path"],
        input_res=configs_ddpm["data"]["image_size"],
    )
    vae.eval()

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

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    # DDPM framework, aka refiner
    ddpm_cls = DDPMv2 if configs_ddpm["training"]["type"] == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=configs_ddpm["model"]["beta1"],
        beta_2=configs_ddpm["model"]["beta2"],
        T=configs_ddpm["model"]["n_timesteps"],
        var_type=configs_ddpm["evaluation"]["variance"],
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=configs_ddpm["model"]["beta1"],
        beta_2=configs_ddpm["model"]["beta2"],
        T=configs_ddpm["model"]["n_timesteps"],
        var_type=configs_ddpm["evaluation"]["variance"],
    )

    # Load pretrained Wrapper function for the whole Diffuse-TreeVAE model
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        configs_ddpm["evaluation"]["chkpt_path"],
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=False if configs_ddpm["training"]["type"] == "uncond" else True,
        pred_steps=configs_ddpm["evaluation"]["n_steps"],
        eval_mode=configs_ddpm["evaluation"]["eval_mode"],
        resample_strategy=configs_ddpm["evaluation"]["resample_strategy"],
        skip_strategy=configs_ddpm["evaluation"]["skip_strategy"],
        sample_method=configs_ddpm["evaluation"]["sample_method"],
        sample_from=configs_ddpm["evaluation"]["sample_from"],
        data_norm=configs_ddpm["data"]["norm"],
        temp=configs_ddpm["evaluation"]["temp"],
        guidance_weight=configs_ddpm["evaluation"]["guidance_weight"],
        z_cond=configs_ddpm["training"]["z_cond"],
        ddpm_latents=ddpm_latents,
        strict=True,
    )

    # Setup callbacks
    write_callback = ImageWriter(
        configs_ddpm["evaluation"]["save_path"],
        "batch",
        n_steps=configs_ddpm["evaluation"]["n_steps"],
        eval_mode=configs_ddpm["evaluation"]["eval_mode"],
        conditional=False if configs_ddpm["training"]["type"] == "uncond" else True,
        sample_prefix=configs_ddpm["evaluation"]["sample_prefix"],
        save_mode=configs_ddpm["evaluation"]["save_mode"],
        save_vae=configs_ddpm["evaluation"]["save_vae"],
        is_norm=configs_ddpm["data"]["norm"],
    )

    test_kwargs = {}
    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = configs_ddpm["evaluation"]["save_path"]

    # Start evaluation
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, gen_test)


if __name__ == "__main__":
    train()