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
        is_norm=False,
        z_cond = False,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "sample_all_leaves", "recons", "recons_all_leaves"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np
        self.z_cond = z_cond

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

        # save reconstructions for all leaves + original for each sample in dataset
        if self.eval_mode == "recons_all_leaves":
            # If conditional, DDPM is conditioned on TreeVAE
            if self.conditional or self.z_cond:
                # get DDPM and VAE samples
                ddpm_samples, vae_samples = prediction

                if self.save_vae:
                    # save TreeVAE samples
                    vae_save_path = os.path.join(self.output_dir, f"vae/{self.eval_mode}")
                    os.makedirs(vae_save_path, exist_ok=True)

                    # send all samples to cpu, vae_samples is a list
                    recons = move_to(vae_samples[0], 'cpu')
                    node_leaves = move_to(vae_samples[1], 'cpu')
                    inputs = batch[0].cpu()
                    labels = batch[1].cpu()
                    num_leaves = len(recons)

                    # loop over each sample in batch
                    for i in range(len(batch_indices)):
                        # save original image from batch
                        fig, axs = plt.subplots(1, num_leaves + 1, figsize=(15, 2))
                        axs[num_leaves].set_title(f"Class: {labels[i].item()}")
                        axs[num_leaves].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
                        axs[num_leaves].set_title("Original")
                        axs[num_leaves].axis('off')
                        # loop over each leaf and save the reconstructed image of the TreeVAE model
                        for c in range(num_leaves):
                            prob = node_leaves[c]['prob'][i]
                            axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                            axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                            axs[c].axis('off')
                        # save whole image
                        plt.savefig(os.path.join(vae_save_path, f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}_{i}.png"))
                        plt.close()
            # If not conditional, DDPM is not conditioned on TreeVAE
            else:
                ddpm_samples = prediction

            # Send all samples to cpu, ddpm_samples is a list for each leaf
            for i in range(len(ddpm_samples)):
                ddpm_samples[i] = move_to(ddpm_samples[i], 'cpu')

            # setup dirs
            base_save_path = os.path.join(self.output_dir, "ddpm")
            img_save_path = os.path.join(base_save_path, "recons_all_l")
            os.makedirs(img_save_path, exist_ok=True)

            # send all samples to cpu, vae_samples is a list
            num_leaves = len(ddpm_samples)
            inputs = batch[0].cpu()
            labels = batch[1].cpu()
            recons = ddpm_samples
            if not self.save_vae:
                node_leaves = None

            # loop over each sample in batch
            for i in range(len(batch_indices)):
                # save original image from batch
                fig, axs = plt.subplots(1, num_leaves + 1, figsize=(15, 2))
                axs[num_leaves].set_title(f"Class: {labels[i].item()}")
                axs[num_leaves].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
                axs[num_leaves].set_title("Original")
                axs[num_leaves].axis('off')
                # loop over each leaf and save the reconstructed image of the DDPM
                for c in range(num_leaves):
                    prob = node_leaves[c]['prob'][i]
                    axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                    axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs[c].axis('off')
                # save whole image
                plt.savefig(os.path.join(img_save_path, f"output_{self.sample_prefix}_{rank}_{batch_idx}_{i}.png"))
                plt.close()

            # loop over each class and save every DDPM reconstruction of this class separately
            for c in range(num_leaves):
                # Setup a dir for each class
                class_save_pass = os.path.join(img_save_path, f"img_cluster_{c}")
                os.makedirs(class_save_pass, exist_ok=True)
                # save every image of this class separately
                for i in range(len(batch_indices)):
                    prob = node_leaves[c]['prob'][i]
                    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
                    axs.imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                    axs.set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs.axis('off')
                    # save image
                    plt.savefig(os.path.join(class_save_pass, f"output_{self.sample_prefix}_{rank}_{batch_idx}_{i}_{prob}.png"))
                    plt.close()

        # save samples for all leaves + original for each sample in dataset
        elif self.eval_mode == "sample_all_leaves":
            # If conditional, DDPM is conditioned on TreeVAE
            if self.conditional or self.z_cond:
                # get DDPM and VAE samples
                ddpm_samples, vae_samples = prediction

                if self.save_vae:
                    # save TreeVAE samples
                    vae_save_path = os.path.join(self.output_dir, f"vae/{self.eval_mode}")
                    os.makedirs(vae_save_path, exist_ok=True)

                    # send all samples to cpu, vae_samples is a list
                    recons = move_to(vae_samples[0], 'cpu')
                    p_c_z = move_to(vae_samples[1], 'cpu')
                    num_leaves = len(recons)

                    # loop over each sample in batch
                    for i in range(len(batch_indices)):
                        # save samples for each leaf
                        if num_leaves == 1:  # needed to avoid an error when plotting only one image
                            fig, axs = plt.subplots(1, 1, figsize=(15, 2))
                            axs.imshow(display_image(recons[0][i]), cmap=plt.get_cmap('gray'))
                            axs.set_title(f"L0: " + f"p=%.2f" % torch.round(p_c_z[i][0], decimals=2))
                            axs.axis('off')
                        else:
                            fig, axs = plt.subplots(1, num_leaves, figsize=(15, 2))
                            for c in range(num_leaves):
                                axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                                axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(p_c_z[i][c], decimals=2))
                                axs[c].axis('off')
                        # save image
                        plt.savefig(os.path.join(vae_save_path, f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}_{i}.png"))
                        plt.close()
            # If not conditional, DDPM is not conditioned on TreeVAE
            else:
                ddpm_samples = prediction

            # send all samples to cpu, ddpm_samples is a list
            for i in range(len(ddpm_samples)):
                ddpm_samples[i] = move_to(ddpm_samples[i], 'cpu')

            # setup dirs
            base_save_path = os.path.join(self.output_dir, "ddpm")
            img_save_path = os.path.join(base_save_path, "sample_all_l")
            os.makedirs(img_save_path, exist_ok=True)

            # ddpm_samples is a list for each leaf
            num_leaves = len(ddpm_samples)
            recons = ddpm_samples

            # loop over each sample in batch
            for i in range(len(batch_indices)):
                # save samples for each leaf
                if num_leaves == 1:  # needed to avoid an error when plotting only one image
                    fig, axs = plt.subplots(1, 1, figsize=(15, 2))
                    axs.imshow(display_image(recons[0][i]), cmap=plt.get_cmap('gray'))
                    axs.set_title(f"L0: " + f"p=%.2f" % torch.round(p_c_z[i][0], decimals=2))
                    axs.axis('off')
                else:
                    fig, axs = plt.subplots(1, num_leaves, figsize=(15, 2))
                    for c in range(num_leaves):
                        axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                        axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(p_c_z[i][c], decimals=2))
                        axs[c].axis('off')
                # save image
                plt.savefig(os.path.join(img_save_path, f"output_{self.sample_prefix}_{rank}_{batch_idx}_{i}.png"))
                plt.close()

            # loop over each class and save every DDPM sample of this class separately
            for c in range(num_leaves):
                # setup a dir for each class
                class_save_pass = os.path.join(img_save_path, f"img_cluster_{c}")
                os.makedirs(class_save_pass, exist_ok=True)
                # save every image of this class separately
                for i in range(len(batch_indices)):
                    prob = p_c_z[i][c]
                    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
                    axs.imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                    axs.set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs.axis('off')
                    # save image
                    plt.savefig(os.path.join(class_save_pass, f"output_{self.sample_prefix}_{rank}_{batch_idx}_{i}_{prob}.png"))
                    plt.close()

        # self.eval_mode in ["sample", "recons"] --> only save samples or reconstructions for selected leaf
        else:
            # If conditional, DDPM is conditioned on TreeVAE
            if self.conditional or self.z_cond:
                # get DDPM and VAE samples
                ddpm_samples_dict, vae_samples = prediction

                if self.save_vae:
                    # save TreeVAE samples
                    vae_samples = vae_samples.cpu()
                    vae_save_path = os.path.join(self.output_dir, f"vae/{self.eval_mode}")
                    os.makedirs(vae_save_path, exist_ok=True)
                    self.save_fn(
                        vae_samples,
                        file_name=os.path.join(
                            vae_save_path,
                            f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                        ),
                        denorm=self.is_norm,
                    )


            # If not conditional, DDPM is not conditioned on TreeVAE
            else:
                ddpm_samples_dict = prediction

            # Save DDPM samples
            for k, ddpm_samples in ddpm_samples_dict.items():
                ddpm_samples = ddpm_samples.cpu()

                # Setup dirs
                img_save_path = os.path.join(self.output_dir, f"ddpm/{self.eval_mode}")
                os.makedirs(img_save_path, exist_ok=True)

                # Save
                self.save_fn(
                    ddpm_samples,
                    file_name=os.path.join(
                        img_save_path, f"output_{self.sample_prefix}_{rank}_{batch_idx}"
                    ),
                    denorm=self.is_norm,
                )
