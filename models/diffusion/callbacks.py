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
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons", "recons_all_leaves"]
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

        if self.eval_mode == "recons_all_leaves":
            if self.conditional:
                ddpm_samples, vae_samples = prediction

                if self.save_vae:
                    vae_save_path = os.path.join(self.output_dir, "vae_all_l")
                    os.makedirs(vae_save_path, exist_ok=True)

                    recons = move_to(vae_samples[0], 'cpu')
                    node_leaves = move_to(vae_samples[1], 'cpu')
                    inputs = batch[0].cpu()
                    labels = batch[1].cpu()
                    num_leaves = len(recons)

                    for i in range(len(batch_indices)):
                        fig, axs = plt.subplots(1, num_leaves + 1, figsize=(15, 2))
                        axs[num_leaves].set_title(f"Class: {labels[i].item()}")
                        axs[num_leaves].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
                        axs[num_leaves].set_title("Original")
                        axs[num_leaves].axis('off')
                        for c in range(num_leaves):
                            prob = node_leaves[c]['prob'][i]
                            axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                            axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                            axs[c].axis('off')
                        # save image
                        plt.savefig(os.path.join(vae_save_path, f"output_vae_{self.sample_prefix}_{rank}_{i}.png"))
                        plt.close()

            else:
                ddpm_samples = prediction


            # send all samples to cpu, ddpm_samples is a list
            for i in range(len(ddpm_samples)):
                ddpm_samples[i] = move_to(ddpm_samples[i], 'cpu')

            # setup dirs
            base_save_path = os.path.join(self.output_dir, "ddpm")
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)

            # save
            num_leaves = len(ddpm_samples)
            inputs = batch[0].cpu()
            labels = batch[1].cpu()
            recons = ddpm_samples
            if not self.save_vae:
                node_leaves = None


            for i in range(len(batch_indices)):
                fig, axs = plt.subplots(1, num_leaves + 1, figsize=(15, 2))
                axs[num_leaves].set_title(f"Class: {labels[i].item()}")
                axs[num_leaves].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
                axs[num_leaves].set_title("Original")
                axs[num_leaves].axis('off')
                for c in range(num_leaves):
                    prob = node_leaves[c]['prob'][i]
                    axs[c].imshow(display_image(recons[c][i]), cmap=plt.get_cmap('gray'))
                    axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(prob, decimals=2))
                    axs[c].axis('off')
                # save image
                plt.savefig(os.path.join(img_save_path, f"output_{self.sample_prefix}_{rank}_{i}.png"))
                plt.close()

            # loop over each class
            for c in range(num_leaves):
                # setup a dir for each class
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
                    plt.savefig(os.path.join(class_save_pass, f"output_{self.sample_prefix}_{rank}_{i}_{prob}.png"))
                    plt.close()

        else:
            if self.conditional:
                ddpm_samples_dict, vae_samples = prediction

                if self.save_vae:
                    vae_samples = vae_samples.cpu()
                    vae_save_path = os.path.join(self.output_dir, "vae")
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

            for k, ddpm_samples in ddpm_samples_dict.items():
                ddpm_samples = ddpm_samples.cpu()

                # Setup dirs
                base_save_path = os.path.join(self.output_dir, k)
                img_save_path = os.path.join(base_save_path, "images")
                os.makedirs(img_save_path, exist_ok=True)

                # Save
                self.save_fn(
                    ddpm_samples,
                    file_name=os.path.join(
                        img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                    ),
                    denorm=self.is_norm,
                )


        # FIXME: This is currently broken. Separate this from the core logic
        # into a new function. Uncomment when ready!
        # if self.compare:
        #     # Save comparisons
        #     (_, img_samples), _ = batch
        #     img_samples = normalize(img_samples).cpu()
        #     iter_ = vae_samples if self.eval_mode == "sample" else img_samples
        #     for idx, (ddpm_pred, pred) in enumerate(zip(ddpm_samples, iter_)):
        #         samples = {
        #             "VAE" if self.eval_mode == "sample" else "Original": pred,
        #             "DDPM": ddpm_pred,
        #         }
        #         compare_samples(
        #             samples,
        #             save_path=os.path.join(
        #                 self.comp_save_path, f"compare_form1_{rank}_{idx}.png"
        #             ),
        #         )
