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
        assert eval_mode in ["sample", "recons", "recons_all_leaves"]
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

        self.vae.eval()

        cond = None
        z = None
        if self.conditional:
            x = batch[0]
            with torch.no_grad():
                # mu, logvar = self.vae.encode(x * 0.5 + 0.5)
                # z = self.vae.reparameterize(mu, logvar)
                # cond = self.vae.decode(z)
                # cond = 2 * cond - 1
                
                # get embeddings z and reconstructions from treevae
                res = self.vae.compute_reconstruction(x)

                max_z_sample = []
                max_recon = []
                leaf_ind = []

                nodes = res[1]
                for i in range(len(nodes[0]['prob'])):
                    probs = [node['prob'][i] for node in nodes]
                    z_sample = [node['z_sample'][i] for node in nodes]
                    if self.max_leaf: # use leaf with max prob
                        ind = probs.index(max(probs))
                    else: # sample one leaf given the probs
                        ind = torch.multinomial(torch.stack(probs), 1).item()
                    max_z_sample.append(z_sample[ind])
                    max_recon.append(res[0][ind][i])
                    leaf_ind.append(ind)

                # z = torch.stack(max_z_sample)
                # use leaf index as conditioning signal, z should be (batch, 1)
                z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(x.device)
                cond = torch.stack(max_recon)
                # cond = 2 * cond - 1


            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
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

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            # recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)

        elif self.eval_mode == "recons_all_leaves":
            img = batch[0]
            recons = self.vae.compute_reconstruction(img)

            # store all refined reconstructions
            out_all_leaves = []

            # same noise for same sample
            noise = torch.randn_like(img)

            # now for each leaf node, we use the recons to condition the ddpm
            for l in range(len(recons[0])):
                recons_leaf_l = recons[0][l]

                # all leaves have the same noise --> reset seeds
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                np.random.seed(0)

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

                if isinstance(self.online_network, DDPMv2):
                    x_t_l += recons_leaf_l

                out = self(
                    x_t_l,
                    cond=recons_leaf_l,
                    z=z if self.z_cond else None,
                    n_steps=self.pred_steps,
                    checkpoints=self.pred_checkpoints,
                    ddpm_latents=self.ddpm_latents,
                )

                out_all_leaves.append(out[str(self.online_network.T)])

            return out_all_leaves, recons


        elif self.eval_mode == "recons":
            img = batch[0]
            recons, z = self.vae.forward_recons(img, self.max_leaf)
            # recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

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
