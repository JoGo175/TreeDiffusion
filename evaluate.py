import copy
import logging
import os
import yaml
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from models.diffusion.callbacks import EMAWeightUpdate, ImageWriter
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import UNetModel, SuperResModel

from models.model import TreeVAE
from utils.diffusion_utils import configure_device
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    return config

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]



def train(model_path):
    config_path = "configs/mnist.yml"
    config = load_config(config_path)
    config_ddpm = config.ddpm

    # Set seed
    #seed_everything(config_ddpm.training.seed, workers=True)

    with open(model_path + "/config.yaml", 'r') as stream:
            configs = yaml.load(stream,Loader=yaml.Loader)

    # Dataset
    trainset, trainset_eval, testset = get_data(configs)
    gen_train = get_gen(trainset, configs, validation=False, shuffle=False)
    gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    gen_train_eval_iter = iter(gen_train_eval)
    gen_test_iter = iter(gen_test)
    y_train = trainset_eval.dataset.targets.clone().detach()[trainset_eval.indices].numpy()
    y_test = testset.dataset.targets.clone().detach()[testset.indices].numpy()

    N = len(trainset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)


    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    checkpoint_path = model_path
    vae = TreeVAE(**configs['training'])
    data_tree = np.load(checkpoint_path+'/data_tree.npy', allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = construct_tree_fromnpy(vae, data_tree, configs)
    if not (configs['globals']['eager_mode'] and configs['globals']['wandb_logging']!='offline'):
        pass
        #model = torch.compile(model)
    vae.load_state_dict(torch.load(checkpoint_path+'/model_weights.pt', map_location=device), strict=True)
    vae.to(device)
    vae.eval()


    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=config_ddpm.data.n_channels,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
        z_dim=config_ddpm.evaluation.z_dim,
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode=config_ddpm.evaluation.eval_mode,
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        ddpm_latents=ddpm_latents,
        strict=True,
    )


    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8


    # Predict trainer
    write_callback = ImageWriter(
        config_ddpm.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode=config_ddpm.evaluation.eval_mode,
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_mode=config_ddpm.evaluation.save_mode,
        save_vae=config_ddpm.evaluation.save_vae,
        is_norm=config_ddpm.data.norm,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path

    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, gen_test)


if __name__ == "__main__":
    model_path = "models/experiments/mnist/20240204-032408_09971"
    train(model_path)






