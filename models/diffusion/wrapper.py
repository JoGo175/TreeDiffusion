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
import torch
import torch.nn as nn
from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from utils.diffusion_utils import space_timesteps
import pytorch_lightning as pl
import numpy as np


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "sample_all_leaves", "recons", "recons_all_leaves"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

        # TreeVAE use max_leaf or sample_leaf
        self.max_leaf = False

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)
            # use Denoising Diffusion Implicit Model sampling
            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        # set the vae to eval mode, no training
        self.vae.eval()

        # conditioning signal and latent z from TreeVAE, cond corresponds to the reconstructions
        cond = None
        z = None
        if self.conditional:
            x = batch[0]
            with torch.no_grad():
                # Compute the reconstructions and the leaf embeddings from the TreeVAE
                res = self.vae.compute_reconstruction(x)
                recons = res[0]
                nodes = res[1]

                # Save the chosen leaf_embeddings, reconstructions and the respective leaf indices
                max_z_sample = []
                max_recon = []
                leaf_ind = []

                # Iterate over the leaf nodes and select the leaf with the highest probability or sample given the leaf probs
                for i in range(len(nodes[0]['prob'])):
                    probs = [node['prob'][i] for node in nodes]
                    z_sample = [node['z_sample'][i] for node in nodes]
                    if self.max_leaf:   # use leaf with max prob
                        ind = probs.index(max(probs))
                    else:               # sample one leaf given the leaf probs
                        ind = torch.multinomial(torch.stack(probs), 1).item()
                    max_z_sample.append(z_sample[ind])
                    max_recon.append(recons[ind][i])
                    leaf_ind.append(ind)

                # z = torch.stack(max_z_sample) # use latent embeddings as conditioning signal
                # here, we use leaf index as conditioning signal instead of latent embeddings, z should be (batch, 1)
                z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(x.device)
                cond = torch.stack(max_recon)


            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            # unconditional, just a normal DDPM
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise at timepoint t
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z if self.z_cond else None
        )

        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # if not conditional --> just a normal DDPM
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        # sample mode --> generate new samples, only uses one leaf
        if self.eval_mode == "sample":
            # From DiffuseVAE:
            # x_t, z = batch
            # recons = self.vae(z)
            # recons = 2 * recons - 1
            # Initial temperature scaling
            # x_t = x_t * self.temp
            # Formulation-2 initial latent
            # if isinstance(self.online_network, DDPMv2):
            #     x_t = recons + self.temp * torch.randn_like(recons)

            # Sample from the TreeVAE
            # instead of using batch of pre-sampled noise as in DiffuseVAE,
            # we resample as many times as there are samples in the Test set to create new samples
            n_samples = batch[0].size(0)
            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            reconstructions, p_c_z = self.vae.generate_images(n_samples, batch[0].device)

            # Save the chosen reconstructions and the respective leaf indices
            max_recon = []
            leaf_ind = []

            # Iterate over the leaf nodes and select the leaf with the highest probability or sample given the leaf probs
            for i in range(len(p_c_z)):
                probs = p_c_z[i]
                if self.max_leaf:
                    ind = torch.argmax(probs)
                else:
                    ind = torch.multinomial(probs, 1).item()

                max_recon.append(reconstructions[ind][i])
                leaf_ind.append(ind)

            # z = torch.stack(max_z_sample) # use latent embeddings as conditioning signal
            # here, we use leaf index as conditioning signal instead of latent embeddings, z should be (batch, 1)
            z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(batch[0].device)
            recons = torch.stack(max_recon)

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                recons,
                torch.randn_like(recons),
                torch.tensor(
                    [self.online_network.T - 1] * recons.size(0), device=recons.device
                ),
            )
            # second formulation for conditioning the forward process, see DiffuseVAE paper
            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        # sample all leaves mode --> generate new samples for each leaf node
        elif self.eval_mode == "sample_all_leaves":
            # Sample from the TreeVAE
            # instead of using batch of pre-sampled noise as in DiffuseVAE,
            # we resample as many times as there are samples in the Test set to create new samples
            n_samples = batch[0].size(0)
            reconstructions, p_c_z = self.vae.generate_images(n_samples, batch[0].device)

            # store all refined reconstructions
            out_all_leaves = []

            # use the same noise for same sample across all leaves
            noise = torch.randn_like(batch[0])

            # sample overall seed to reset seeds for each leaf,
            # thus, each leaf will have the same noise for the same sample and
            # only differ in the reconstructions and conditioning signal, given by each leaf in TreeVAE
            seed_val = np.random.randint(0, 1000)

            # now for each leaf node, we use the recons to condition the ddpm
            for l in range(len(reconstructions)):
                recons_leaf_l = reconstructions[l]

                # all leaves have the same noise --> reset seeds
                torch.manual_seed(seed_val)
                torch.cuda.manual_seed(seed_val)
                np.random.seed(seed_val)

                # z is the leaf index
                z = torch.tensor([l]*n_samples, dtype=torch.float).unsqueeze(1).to(batch[0].device)

                # DDPM encoder
                x_t_l = self.online_network.compute_noisy_input(
                    recons_leaf_l,
                    noise,
                    torch.tensor(
                        [self.online_network.T - 1] * recons_leaf_l.size(0), device=recons_leaf_l.device
                    ),
                )
                # second formulation for conditioning the forward process, see DiffuseVAE paper
                if isinstance(self.online_network, DDPMv2):
                    x_t_l += recons_leaf_l

                # sample from the DDPM given the conditioning signal and the reconstructions
                out = self(
                    x_t_l,
                    cond=recons_leaf_l,
                    z=z if self.z_cond else None,
                    n_steps=self.pred_steps,
                    checkpoints=self.pred_checkpoints,
                    ddpm_latents=self.ddpm_latents,
                )
                # save the samples for each leaf
                out_all_leaves.append(out[str(self.online_network.T)])
            return out_all_leaves, (reconstructions, p_c_z)

        # recons mode --> refine the data reconstructions from the TreeVAE
        elif self.eval_mode == "recons":
            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            img = batch[0]
            recons, z = self.vae.forward_recons(img, self.max_leaf)

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )
            # second formulation for conditioning the forward process, see DiffuseVAE paper
            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        # recons all leaves mode --> refine the data reconstructions from the TreeVAE for each leaf node
        elif self.eval_mode == "recons_all_leaves":
            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            img = batch[0]
            recons = self.vae.compute_reconstruction(img)

            # store all refined reconstructions
            out_all_leaves = []

            # same noise for same sample
            noise = torch.randn_like(img)

            # sample overall seed to reset seeds for each leaf,
            # thus, each leaf will have the same noise for the same sample and
            # only differ in the reconstructions and conditioning signal, given by each leaf in TreeVAE
            seed_val = np.random.randint(0, 1000)

            # now for each leaf node, we use the recons and the conditioning signal to condition the ddpm
            for l in range(len(recons[0])):
                recons_leaf_l = recons[0][l]

                # all leaves have the same noise --> reset seeds
                torch.manual_seed(seed_val)
                torch.cuda.manual_seed(seed_val)
                np.random.seed(seed_val)

                # z is the leaf index
                z = torch.tensor([l]*img.size(0), dtype=torch.float).unsqueeze(1).to(img.device)

                # DDPM encoder
                x_t_l = self.online_network.compute_noisy_input(
                    img,
                    noise,
                    torch.tensor(
                        [self.online_network.T - 1] * img.size(0), device=img.device
                    ),
                )
                # second formulation for conditioning the forward process, see DiffuseVAE paper
                if isinstance(self.online_network, DDPMv2):
                    x_t_l += recons_leaf_l

                # sample from the DDPM given the conditioning signal and the reconstructions
                out = self(
                    x_t_l,
                    cond=recons_leaf_l,
                    z=z if self.z_cond else None,
                    n_steps=self.pred_steps,
                    checkpoints=self.pred_checkpoints,
                    ddpm_latents=self.ddpm_latents,
                )
                # save the samples for each leaf
                out_all_leaves.append(out[str(self.online_network.T)])
            return out_all_leaves, recons

        # For eval_mode in ["sample", "recons"]:
        # Given the reconstructions and the conditioning signal from the TreeVAE,
        # we use the DDPM to refine the reconstructions or the new generated samples
        out = (
            self(
                x_t,
                cond=recons,
                z=z if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )
        return out

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )
        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
