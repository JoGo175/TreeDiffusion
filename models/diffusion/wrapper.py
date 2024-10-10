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
import gc
import concurrent.futures


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
        z_signal="cluster_id"
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

        # Conditioning on cluster_id or latent embeddings
        self.z_signal = z_signal

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

        # condition on reconstructions (and z) from TreeVAE
        if self.conditional or self.z_cond:
            x = batch[0]
            with torch.no_grad():
                # Compute the reconstructions and the leaf embeddings from the TreeVAE
                res = self.vae.forward(x)
                recons = res['reconstructions']
                nodes = res['node_leaves']
                node_info = res['node_info']

                vae_recon, cond, z = self.compute_reconstructions_and_leaf_embeddings(recons, nodes, node_info,
                                                                                      self.max_leaf,
                                                                                      self.z_signal,
                                                                                      self.conditional, x.device)

            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)

        # unconditional, just a normal DDPM
        else:
            x = batch[0]

        # Clear unused memory
        _ = gc.collect()
        torch.cuda.empty_cache()

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

    def get_path_info(self, node_info_list, selected_leaf_indices):
        def traverse_to_root_iterative(start_node, node_list, sample_index=None):
            path_info = []
            node_ids = []
            z_samples = []
            current_node = start_node

            # Iteratively traverse to the root
            while current_node['parent_id'] is not None:
                node_ids.append(current_node['node_id'])  # Keep node_id as integer
                z_samples.append(current_node['z_sample'][sample_index])
                current_node = node_list[current_node['parent_id']]

            # Append the root node info
            node_ids.append(current_node['node_id'])
            z_samples.append(current_node['z_sample'][sample_index])

            # Convert to tensors in one go and move to device
            node_ids_tensor = torch.tensor(node_ids, dtype=torch.float).to(self.device)
            z_samples_tensor = torch.stack(z_samples).to(self.device)  # Stack z_samples into a tensor

            # Combine the tensors into path_info
            path_info = [(nid, z) for nid, z in zip(node_ids_tensor, z_samples_tensor)]

            return path_info

        def process_leaf(l_leaf):
            l, leaf_index = l_leaf
            node_info = node_info_dict.get(leaf_index)

            if node_info:
                return traverse_to_root_iterative(node_info, node_info_list, l)
            return None

        # Preprocess node_info_list into a dictionary for fast lookup by leaf_id
        node_info_dict = {node_info['leaf_id']: node_info for node_info in node_info_list}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_leaf, enumerate(selected_leaf_indices)))

        output = [r for r in results if r is not None]
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        # set the vae to eval mode, no training
        self.vae.eval()

        # conditioning signal and latent z from TreeVAE, cond corresponds to the reconstructions
        cond = None
        z = None

        # -----------------------------------------------------------------------------------------------
        # if not conditional --> just a normal DDPM
        if not self.conditional and not self.z_cond:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )

            if self.eval_mode == "sample":
                x_t = torch.randn_like(batch[0])

            # for DDPM, recons mode is the same as sample mode because we don't have any conditioning signal
            elif self.eval_mode == "recons":
                img = batch[0]
                # DDPM encoder
                x_t = self.online_network.compute_noisy_input(
                    img,
                    torch.randn_like(img),
                    torch.tensor(
                        [self.online_network.T - 1] * img.size(0), device=img.device
                    ),
                )

            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        # -----------------------------------------------------------------------------------------------
        # conditional --> use the reconstructions from the TreeVAE as conditioning signal

        # sample mode --> generate new samples, only uses one leaf
        if self.eval_mode == "sample":
            # Sample from the TreeVAE
            # instead of using batch of pre-sampled noise as in DiffuseVAE,
            # we resample as many times as there are samples in the Test set to create new samples
            n_samples = batch[0].size(0)
            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            recons, nodes, _, node_info = self.vae.generate_images_and_embeddings(n_samples, batch[0].device)
            vae_recon, cond, z = self.compute_reconstructions_and_leaf_embeddings(recons, nodes, node_info, self.max_leaf,
                                                                       self.z_signal, self.conditional, batch[0].device)

            # Sample DDPM latent noise
            x_t = torch.randn_like(batch[0])

            # second formulation for conditioning the forward process, see DiffuseVAE paper
            if isinstance(self.online_network, DDPMv2):
                x_t += cond

        # sample all leaves mode --> generate new samples for each leaf node
        elif self.eval_mode == "sample_all_leaves":
            # Sample from the TreeVAE
            # instead of using batch of pre-sampled noise as in DiffuseVAE,
            # we resample as many times as there are samples in the Test set to create new samples
            n_samples = batch[0].size(0)
            reconstructions, nodes, p_c_z, node_info = self.vae.generate_images_and_embeddings(n_samples, batch[0].device)

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

                # leaf index + latent embeddings as conditioning signal
                if self.z_signal == "both":
                    z1 = nodes[l]['z_sample']
                    z2 = torch.tensor([l]*n_samples, dtype=torch.float).unsqueeze(1).to(batch[0].device)
                    z = [z1, z2]

                # latent embeddings as conditioning signal
                elif self.z_signal == "latent":
                    z = nodes[l]['z_sample']

                # leaf index as conditioning signal instead of latent embeddings, z should be (batch, 1)
                elif self.z_signal == "cluster_id":
                    z = torch.tensor([l]*n_samples, dtype=torch.float).unsqueeze(1).to(batch[0].device)

                elif self.z_signal == "path":  # full path
                    z = self.get_path_info(node_info, [l]*n_samples)

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

                if not self.conditional:
                    recons_leaf_l = None

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
                out_all_leaves.append(next(iter(out.values())))
            return out_all_leaves, (reconstructions, p_c_z)

        # recons mode --> refine the data reconstructions from the TreeVAE
        elif self.eval_mode == "recons":

            img = batch[0]

            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            recons, nodes, node_info = self.vae.compute_reconstruction(img)
            vae_recon, cond, z = self.compute_reconstructions_and_leaf_embeddings(recons, nodes, node_info, self.max_leaf,
                                                                       self.z_signal, self.conditional, img.device)

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
                x_t += cond

        # recons all leaves mode --> refine the data reconstructions from the TreeVAE for each leaf node
        elif self.eval_mode == "recons_all_leaves":
            # Compute the reconstructions and the leaf embeddings from the TreeVAE
            img = batch[0]
            recons, nodes, node_info = self.vae.compute_reconstruction(img)

            # store all refined reconstructions
            out_all_leaves = []

            # same noise for same sample
            noise = torch.randn_like(img)

            # sample overall seed to reset seeds for each leaf,
            # thus, each leaf will have the same noise for the same sample and
            # only differ in the reconstructions and conditioning signal, given by each leaf in TreeVAE
            seed_val = np.random.randint(0, 1000)

            # now for each leaf node, we use the recons and the conditioning signal to condition the ddpm
            for l in range(len(recons)):
                recons_leaf_l = recons[l]
                nodes_leaf_l = nodes[l]

                # all leaves have the same noise --> reset seeds
                torch.manual_seed(seed_val)
                torch.cuda.manual_seed(seed_val)
                np.random.seed(seed_val)

                # leaf index + latent embeddings as conditioning signal
                if self.z_signal == "both":
                    z1 = nodes_leaf_l['z_sample']
                    z2 = torch.tensor([l]*img.size(0), dtype=torch.float).unsqueeze(1).to(img.device)
                    z = [z1, z2]

                # latent embeddings as conditioning signal
                elif self.z_signal == "latent":
                    z = nodes_leaf_l['z_sample']

                # leaf index as conditioning signal instead of latent embeddings, z should be (batch, 1)
                elif self.z_signal == "cluster_id":
                    z = torch.tensor([l]*img.size(0), dtype=torch.float).unsqueeze(1).to(img.device)

                elif self.z_signal == "path":  # full path
                    z = self.get_path_info(node_info, [l]*img.size(0))

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

                if not self.conditional:
                    recons_leaf_l = None

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
                out_all_leaves.append(next(iter(out.values())))
            return out_all_leaves, (recons, nodes)

        # For eval_mode in ["sample", "recons"]:
        # Given the reconstructions and the conditioning signal from the TreeVAE,
        # we use the DDPM to refine the reconstructions or the new generated samples
        out = (
            self(
                x_t,
                cond=cond,
                z=z if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            vae_recon,
        )
        return out


    def compute_reconstructions_and_leaf_embeddings(self, recons, nodes, node_info, max_leaf=False, z_signal="both",
                                                    conditional=False, device='cpu'):
        """
        Function to compute reconstructions and leaf embeddings from the TreeVAE model.

        Args:
            recons (Tensor): Tensor containing the reconstructions from the TreeVAE.
            nodes (list of dicts): List containing information for the leaf nodes, including probabilities and latent embeddings.
            node_info (dict): Information about ALL the nodes
            max_leaf (bool): Flag indicating whether to select the leaf with the maximum probability. Default is False.
            z_signal (str): Determines the conditioning signal to use. Options are "latent", "both", "cluster_id", or "path". Default is "both".
            conditional (bool): Flag indicating if the model is conditional. Default is False.
            device (str): The device (e.g., 'cpu' or 'cuda') on which to perform the calculations. Default is 'cpu'.

        Returns:
            z (Tensor or list): The latent embeddings and/or leaf indices used as conditioning signal.
            recons (Tensor): The selected reconstructions based on leaf embeddings.
            leaf_ind (list): The list of leaf indices chosen for the batch.
        """

        # Initialize cond and z
        z = None

        # Save the chosen leaf_embeddings, reconstructions, and the respective leaf indices
        max_z_sample = []
        max_recon = []
        leaf_ind = []

        # Iterate over the leaf nodes and select the leaf with the highest probability or sample given the leaf probs
        for i in range(len(nodes[0]['prob'])):
            probs = [node['prob'][i] for node in nodes]
            z_sample = [node['z_sample'][i] for node in nodes]

            if max_leaf:
                # Select the leaf with the highest probability
                ind = probs.index(max(probs))
            else:
                # Sample one leaf given the leaf probabilities
                ind = torch.multinomial(torch.stack(probs), 1).item()

            max_z_sample.append(z_sample[ind])
            max_recon.append(recons[ind][i])
            leaf_ind.append(ind)

        # Select the appropriate conditioning signal
        if z_signal == "both": # leaf index + latent embeddings
            z1 = torch.stack(max_z_sample)
            z2 = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(device)
            z = [z1, z2]

        elif z_signal == "latent": # latent embeddings
            z = torch.stack(max_z_sample)

        elif z_signal == "cluster_id": # leaf index
            z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(device)

        elif z_signal == "path": # full path
            z = self.get_path_info(node_info, leaf_ind)

        # Select the final reconstructions if we condition on them too
        vae_recon = torch.stack(max_recon)
        if conditional:
            cond = vae_recon
        else:
            cond = None

        return vae_recon, cond, z


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
