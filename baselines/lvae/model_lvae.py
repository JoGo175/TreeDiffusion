"""
TreeVAE model.
"""
import torch
import torch.nn as nn
import torch.distributions as td

from utils.model_utils import compute_posterior
from models.networks import (get_encoder, get_decoder, MLP, Dense)
from utils.training_utils import calc_aug_loss
from baselines.lvae.losses import loss_reconstruction_binary, loss_reconstruction_mse

class LadderVAE(nn.Module):
    def __init__(self, **kwargs):
        super(LadderVAE, self).__init__()
        self.kwargs = kwargs

        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        
        self.alpha=torch.tensor(self.kwargs['kl_start'])

        # saving important variables to initialize the tree
        self.encoded_sizes = self.kwargs['latent_dim']
        self.hidden_layers = self.kwargs['mlp_layers']
        # check that the number of layers for bottom up is equal to top down
        if len(self.encoded_sizes) != len(self.hidden_layers):
            raise ValueError('Model is mispecified!!')
        self.depth = self.kwargs['initial_depth']
        self.inp_shape = self.kwargs['inp_shape']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']

        # bottom up: the inference chain that from x computes the d units until the root
        if self.activation == "mse":
            size = int((self.inp_shape / 3)**0.5)
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.encoded_sizes[0],
                                x_shape=self.inp_shape, size=size)
        else:
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.encoded_sizes[0],
                                x_shape=self.inp_shape)   
            
        self.bottom_up = nn.ModuleList([encoder])
        for i in range(1, len(self.hidden_layers)):
            self.bottom_up.append(MLP(self.hidden_layers[i-1], self.encoded_sizes[i], self.hidden_layers[i]))

        # MLP's if we use contrastive loss on bottom-up embeddings
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']])>0:
            self.contrastive_mlp = nn.ModuleList([])
            for i in range(0, len(self.hidden_layers)):
                self.contrastive_mlp.append(MLP(input_size=self.hidden_layers[i], encoded_size=self.encoded_sizes[i], hidden_unit=min(self.hidden_layers)))

        # select the top down generative networks
        encoded_size_gen = self.encoded_sizes[-(self.depth+1):] # e.g. encoded_sizes 32,16,8, depth 1
        encoded_size_gen = encoded_size_gen[::-1] # encoded_size_gen = 16,8 => 8,16
        encoded_size_gen = encoded_size_gen[1:] # encoded_size_gen = 16 (root does not have a transformation)
        layers_gen = self.hidden_layers[-(self.depth+1):] # e.g. encoded_sizes 256,128,64, depth 1
        layers_gen = layers_gen[::-1] # encoded_size_gen = 128,64 => 64,128
        layers_gen = layers_gen[:-1] # 64 as the leaves have decoder

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = nn.ModuleList([None])
        self.denses = nn.ModuleList([Dense(layers_gen[-1], self.encoded_sizes[-1])]) # the dense layer has latent dim = the dim of the node
        for i in range(self.depth):
                self.transformations.append(MLP(encoded_size_gen[i], encoded_size_gen[i], layers_gen[i]))
                self.denses.append(Dense(layers_gen[i], encoded_size_gen[i]))

        self.decoder = get_decoder(architecture=self.kwargs['encoder'], input_shape=encoded_size_gen[-1],
                                   output_shape=self.inp_shape, activation=self.activation)

    def forward(self, x):
        epsilon = 1e-7  
        device = x.device

        # compute deterministic bottom up
        d = x
        encoders = []
        emb_contr = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d)

            # Pass through contrastive MLP's
            if 'instancewise_full' in self.augmentation_method:
                _, emb_c, _ = self.contrastive_mlp[i](d)
                emb_contr.append(emb_c)
            elif 'instancewise_first' in self.augmentation_method:
                if i==0:
                    _, emb_c, _ = self.contrastive_mlp[i](d)
                    emb_contr.append(emb_c)

            # store bottom-up embeddings for top-down
            encoders.append(d)

        # initializate KL losses
        kl_nodes_tot = torch.zeros(len(x), device=device)
        aug_decisions_loss = torch.zeros(1, device=device)
        depth_level = 0
        z_parent_sample = None
        for i in range(self.depth+1):

            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = self.denses[depth_level](d)

            if depth_level == 0:  
                # here we are in the root
                # standard gaussian
                z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat, device=device), torch.ones_like(z_sigma_q_hat, device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1)
                # sampled z is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = self.transformations[depth_level](z_parent_sample)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # trick used to save the sample from distr in pytorch but basically is same as above
            # mu_var_q = torch.cat([z_mu_q, torch.sqrt(z_sigma_q + epsilon)], dim=1)
            # l = z_mu_q.shape[1]
            # z_q_dist = distributions.transformed_distribution.TransformedDistribution(
            #     distributions.Independent(distributions.Normal(loc=mu_var_q[:, :l], scale=mu_var_q[:, l:]), 1),
            #     [distributions.transforms.AffineTransform(0, 1)]
            # )

            z_parent_sample = z_sample

            # compute KL node
            kl_node = torch.clamp(td.kl_divergence(z, z_p), min=-1, max=10E10)

            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            if depth_level == self.depth:
                reconstruction = self.decoder(z_sample)
            depth_level += 1

        rec_losses = self.loss(x, reconstruction)
        kl_nodes_loss = torch.clamp(torch.mean(kl_nodes_tot), min=-10, max=1e10)
        kl_root_loss = torch.mean(kl_root)
        rec_loss = torch.mean(rec_losses)

        if self.training is True and self.augment and 'simple' not in self.augmentation_method:
            aug_decisions_loss += calc_aug_loss(prob_parent=torch.ones(x.size(0)), prob_router=torch.zeros(x.size(0)), augmentation_methods=self.augmentation_method, emb_contr=emb_contr)

        return {
            'rec': reconstruction,
            'rec_loss': rec_loss,
            'kl_root': kl_root_loss,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'z_leaves': z_sample,
            'elbo_samples': kl_nodes_tot + kl_root + rec_losses
        }