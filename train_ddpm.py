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

from models.diffusion.callbacks import EMAWeightUpdate
from models.diffusion.ddpm import DDPM
from models.diffusion.ddpm_form2 import DDPMv2
from models.diffusion.wrapper import DDPMWrapper
from models.diffusion.unet_openai import UNetModel, SuperResModel

from models.model import TreeVAE
from utils.diffusion_utils import configure_device
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_tree_fromnpy

logger = logging.getLogger(__name__)

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    return config


def train(model_path):
    # Get config and setup
    config_path = "configs/cifar10.yml"
    config = load_config(config_path)
    config = config.ddpm

    with open(model_path + "/config.yaml", 'r') as stream:
        configs = yaml.load(stream,Loader=yaml.Loader)

    # config = ddpm_config
    # configs = model_config for TreeVAE

    logger.info(f"Running with config: {config}")

    # Set seed
    #seed_everything(config.training.seed, workers=True)

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

    # Model
    lr = config.training.lr
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    ddpm_type = config.training.type

    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        num_heads=config.model.n_heads,
        z_dim=config.training.z_dim,
        use_scale_shift_norm=config.training.z_cond,
        use_z=config.training.z_cond,
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )

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

    # Freeze all parameters of VAE
    for p in vae.parameters():
        p.requires_grad = False

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")

    ddpm_wrapper = DDPMWrapper(
        online_ddpm,
        target_ddpm,
        vae,
        lr=lr,
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device
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
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        trainset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config.training.grad_clip

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloaders=gen_train)
    

if __name__ == "__main__":
    model_path = "models/experiments/cifar10/20240103-165129_bc018"
    train(model_path)