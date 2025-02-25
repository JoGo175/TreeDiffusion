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
import os
from typing import Sequence, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from torch.nn import Module
from utils.diffusion_utils import save_as_images, save_as_np
import matplotlib.pyplot as plt
from utils.training_utils import move_to
from utils.utils import display_image


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx = None,
    ) -> None:
        # get networks
        online_net = pl_module.online_network.decoder
        target_net = pl_module.target_network.decoder

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


class ImageWriter(BasePredictionWriter):
    """
    Image writer to save images to disk during evaluation.

    Args:
        output_dir:     Directory to save images to.
        write_interval: Interval to save images at.
        compare:        If True, save both the original and reconstructed images.
        n_steps:        Number of steps to save images for.
        eval_mode:      Evaluation mode to use. One of "sample", "sample_all_leaves", "recons", "recons_all_leaves".
        conditional:    If True, the model is conditional --> DDPM conditional on TreeVAE.
        sample_prefix:  Prefix to use when saving samples.
        save_vae:       If True, save VAE samples.
        save_mode:      Save mode to use. One of "image", "np".
        is_norm:        If True, normalize the images before saving.
    """
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        save_mode="image",
        is_norm=True,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank

        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction

            if self.save_vae:
                vae_samples = vae_samples.cpu()
                base_save_path = os.path.join(self.output_dir, self.eval_mode)
                vae_save_path = os.path.join(base_save_path, "vae")
                os.makedirs(vae_save_path, exist_ok=True)
                self.save_fn(
                    vae_samples,
                    file_name=os.path.join(
                        vae_save_path,
                        f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                    ),
                    denorm=self.is_norm,
                )
        else:
            ddpm_samples_dict = prediction

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        for k, ddpm_samples in ddpm_samples_dict.items():
            ddpm_samples = ddpm_samples.cpu()

            # Setup dir
            base_save_path = os.path.join(self.output_dir, self.eval_mode)
            img_save_path = os.path.join(base_save_path, "ddpm")
            os.makedirs(img_save_path, exist_ok=True)

            # Save
            self.save_fn(
                ddpm_samples,
                file_name=os.path.join(
                    img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                ),
                denorm=self.is_norm,
            )