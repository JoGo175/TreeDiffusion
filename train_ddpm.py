"""

"""
import copy
import os
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from models.diffusion.callbacks import EMAWeightUpdate
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import UNetModel, SuperResModel
from models.model import TreeVAE
from utils.diffusion_utils import configure_device
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy
from utils.utils import reset_random_seeds, prepare_config

###############################################################################################################
# SELECT THE DATASET
dataset = "mnist"       # mnist, fmnist, cifar10, celeba is supported
###############################################################################################################


def train():
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    # Get config and setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=f'{dataset}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba'],
                        help='the override file name for config.yml')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    # Configs specific to DDPM
    configs_ddpm = configs['ddpm']

    # Reproducibility
    reset_random_seeds(configs['globals']['seed'])

    # Dataset
    trainset, trainset_eval, testset = get_data(configs)
    gen_train = get_gen(trainset, configs, validation=False, shuffle=False)

    # Model
    model_path = configs_ddpm["training"]["vae_chkpt_path"]
    attn_resolutions = __parse_str(configs_ddpm["model"]["attn_resolutions"])
    dim_mults = __parse_str(configs_ddpm["model"]["dim_mults"])
    ddpm_type = configs_ddpm["training"]["type"]

    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=configs_ddpm["data"]["n_channels"],
        model_channels=configs_ddpm["model"]["dim"],
        out_channels=configs_ddpm["data"]["n_channels"],
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

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

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

    vae = TreeVAE(**configs['training'])
    data_tree = np.load(model_path+'/data_tree.npy', allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = construct_tree_fromnpy(vae, data_tree, configs)
    if not (configs['globals']['eager_mode'] and configs['globals']['wandb_logging']!='offline'):
        pass
    vae.load_state_dict(torch.load(model_path+'/model_weights.pt', map_location=device), strict=True)
    vae.to(device)
    vae.eval()

    # Freeze all parameters of VAE
    for p in vae.parameters():
        p.requires_grad = False

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)

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

    # Trainer
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

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = configs_ddpm["training"]["epochs"]
    train_kwargs["log_every_n_steps"] = configs_ddpm["training"]["log_step"]
    train_kwargs["callbacks"] = [chkpt_callback]

    if configs_ddpm["training"]["use_ema"]:
        ema_callback = EMAWeightUpdate(tau=configs_ddpm["training"]["ema_decay"])
        train_kwargs["callbacks"].append(ema_callback)

    device = configs_ddpm["training"]["device"]
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        #from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
        from pytorch_lightning.strategies.ddp import DDPStrategy

        #train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        train_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if configs_ddpm["training"]["fp16"]:
        train_kwargs["precision"] = 16

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config.training.grad_clip


    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloaders=gen_train)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]



if __name__ == "__main__":
    train()