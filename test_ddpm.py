import copy
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
from pathlib import Path

from models.diffusion.callbacks import ImageWriter
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import SuperResModel
from models.model import TreeVAE
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds, prepare_config

###############################################################################################################
# SELECT THE DATASET
dataset = "mnist"       # mnist, fmnist, cifar10, celeba is supported
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
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--vae_chkpt_path', default='', type=str, help='path to the pretrained TreeVAE model')
    parser.add_argument('--results_dir', default='', type=str, help='path to the results directory')
    parser.add_argument('--chkpt_path', default='', type=str, help='path to the pretrained DDPM model')
    parser.add_argument('--save_path', default='', type=str, help='path to save the results')
    parser.add_argument('--eval_mode', default='sample', type=str, help='evaluation mode: sample or recons')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    # Configs specific to DDPM
    configs_ddpm = configs['ddpm']
    configs_ddpm['globals']['seed'] = args.seed
    configs_ddpm['evaluation']['eval_mode'] = args.eval_mode
    if args.vae_chkpt_path != '':
        configs_ddpm['training']['vae_chkpt_path'] = args.vae_chkpt_path
    if args.results_dir != '':
        configs_ddpm['training']['results_dir'] = args.results_dir
    if args.chkpt_path != '':
        configs_ddpm['evaluation']['chkpt_path'] = args.chkpt_path
    if args.save_path != '':
        configs_ddpm['evaluation']['save_path'] = args.save_path

    # Reproducibility
    reset_random_seeds(configs_ddpm['globals']['seed'])

    # Dataset
    trainset, trainset_eval, testset = get_data(configs_ddpm)
    gen_test = get_gen(testset, configs_ddpm, validation=True, shuffle=False)

    # Pre-sampled latents for DDPM if available
    ddpm_latent_path = configs_ddpm["data"]["ddpm_latent_path"]
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained TreeVAE model, aka generator
    model_path = configs_ddpm["training"]["vae_chkpt_path"]
    vae = TreeVAE(**configs['training'])
    data_tree = np.load(model_path+'/data_tree.npy', allow_pickle=True)
    vae = construct_tree_fromnpy(vae, data_tree, configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.load_state_dict(torch.load(model_path+'/model_weights.pt', map_location=device), strict=False)
    vae.to(device)
    vae.eval()

    # UNet Denoising Model for DDPM
    attn_resolutions = __parse_str(configs_ddpm["model"]["attn_resolutions"])
    dim_mults = __parse_str(configs_ddpm["model"]["dim_mults"])
    decoder = SuperResModel(
        in_channels=configs_ddpm["data"]["inp_channels"],
        model_channels=configs_ddpm["model"]["dim"],
        out_channels=configs_ddpm["data"]["inp_channels"],
        num_res_blocks=configs_ddpm["model"]["n_residual"],
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=configs_ddpm["model"]["dropout"],
        num_heads=configs_ddpm["model"]["n_heads"],
        z_dim=configs_ddpm["evaluation"]["z_dim"],
        use_scale_shift_norm=configs_ddpm["evaluation"]["z_cond"],
        use_z=configs_ddpm["evaluation"]["z_cond"],
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    # DDPM framework, aka refiner
    ddpm_cls = DDPMv2 if configs_ddpm["evaluation"]["type"] == "form2" else DDPM
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
        conditional=True,
        pred_steps=configs_ddpm["evaluation"]["n_steps"],
        eval_mode=configs_ddpm["evaluation"]["eval_mode"],
        resample_strategy=configs_ddpm["evaluation"]["resample_strategy"],
        skip_strategy=configs_ddpm["evaluation"]["skip_strategy"],
        sample_method=configs_ddpm["evaluation"]["sample_method"],
        sample_from=configs_ddpm["evaluation"]["sample_from"],
        data_norm=configs_ddpm["data"]["norm"],
        temp=configs_ddpm["evaluation"]["temp"],
        guidance_weight=configs_ddpm["evaluation"]["guidance_weight"],
        z_cond=configs_ddpm["evaluation"]["z_cond"],
        ddpm_latents=ddpm_latents,
        strict=True,
    )

    # Setup callbacks
    write_callback = ImageWriter(
        configs_ddpm["evaluation"]["save_path"],
        "batch",
        n_steps=configs_ddpm["evaluation"]["n_steps"],
        eval_mode=configs_ddpm["evaluation"]["eval_mode"],
        conditional=True,
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






