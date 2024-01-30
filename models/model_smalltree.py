"""
Small TreeVAE model.
"""
import torch
import torch.nn as nn
import torch.distributions as td
from models.networks import get_decoder, MLP, Router, Dense, Conv
from utils.model_utils import compute_posterior
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.training_utils import calc_aug_loss

class SmallTreeVAE(nn.Module):
    def __init__(self, depth, **kwargs):
        super(SmallTreeVAE, self).__init__()
        self.kwargs = kwargs
        
        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        # KL-annealing weight initialization
        self.alpha=self.kwargs['kl_start'] 

        # dropout in router
        self.dropout_router = self.kwargs['dropout_router']
        # skip connection in the MLPs of the transformations
        self.skip_con_transformation = self.kwargs['skip_con_transformation']

        latent_channels = self.kwargs['latent_channels']
        bottom_up_channels = self.kwargs['bottom_up_channels']
        representation_dim = self.kwargs['representation_dim']
        self.depth = depth
        self.latent_channel = latent_channels[-self.depth-1]
        self.bottom_up_channel = bottom_up_channels[-self.depth]
        self.inp_shape = self.kwargs['inp_shape']
        self.inp_channel = self.kwargs['inp_channels']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']

        self.denses = nn.ModuleList([Dense(self.bottom_up_channel, self.latent_channel) for _ in range(2)])
        self.transformations = nn.ModuleList([Conv(self.latent_channel, self.latent_channel, skip_connection=self.skip_con_transformation) for _ in range(2)])
        self.decision = Router(self.latent_channel, rep_dim=representation_dim, hidden_units=self.bottom_up_channel, dropout=self.dropout_router)
        self.decision_q = Router(self.bottom_up_channel, rep_dim=representation_dim, hidden_units=self.bottom_up_channel, dropout=self.dropout_router)
        self.decoders = nn.ModuleList([get_decoder(architecture=self.kwargs['encoder'], input_shape=representation_dim,
                                                   input_channels=self.latent_channel, output_shape=int((self.inp_shape)**0.5),
                                                   output_channels=self.inp_channel, activation=self.activation) for _ in range(2)])

    def forward(self, x, z_parent, p, bottom_up):
        epsilon = 1e-7  # Small constant to prevent numerical instability
        device = x.device
        
        # Extract relevant bottom-up
        d_q = bottom_up[-self.depth]
        d = bottom_up[-self.depth - 1]
        
        prob_child_left = self.decision(z_parent).squeeze()
        prob_child_left_q = self.decision_q(d_q).squeeze()
        leaves_prob = [p * prob_child_left_q, p * (1 - prob_child_left_q)]

        kl_decisions = prob_child_left_q * torch.log(epsilon + prob_child_left_q / (prob_child_left + epsilon)) +\
                        (1 - prob_child_left_q) * torch.log(epsilon + (1 - prob_child_left_q) /
                                                                (1 - prob_child_left + epsilon))
        kl_decisions = torch.mean(p * kl_decisions)
        
        # Contrastive loss
        aug_decisions_loss = torch.zeros(1, device=device)
        if self.training is True and self.augment is True and 'simple' not in self.augmentation_method:
            aug_decisions_loss += calc_aug_loss(prob_parent=p, prob_router=prob_child_left_q,
                                                augmentation_methods=self.augmentation_method)

        reconstructions = []
        kl_nodes = torch.zeros(1, device=device)
        decoder_magnitudes = []
        for i in range(2):
            # Compute posterior parameters
            z_mu_q_hat, z_sigma_q_hat = self.denses[i](d)
            _, z_mu_p, z_sigma_p = self.transformations[i](z_parent)
            z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p+epsilon)), 3)
            z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # Compute sample z using mu_q and sigma_q
            z_q = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 3)
            z_sample = z_q.rsample()

            # Compute KL node
            kl_node = torch.mean(leaves_prob[i] * td.kl_divergence(z_q, z_p))
            kl_nodes += kl_node

            reconstructions.append(self.decoders[i](z_sample))

            # get decoder weights
            params = []
            for param in self.decoders[i].parameters():
                if param.requires_grad:
                    params.append(param)
            params = torch.cat([param.view(-1) for param in params])
            decoder_magnitudes.append(torch.norm(params, p=2))

        kl_nodes_loss = torch.clamp(kl_nodes, min=-10, max=1e10)

        # Probability of falling in each leaf
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)    

        return {
            'rec_loss': rec_loss,
            'weights': leaves_prob,
            'kl_decisions': kl_decisions,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
            'decoder_magnitudes': decoder_magnitudes
        }
