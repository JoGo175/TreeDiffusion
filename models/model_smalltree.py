"""
SmallTreeVAE model (used for the growing procedure of TreeVAE).
"""
import torch
import torch.nn as nn
import torch.distributions as td
from models.networks import get_decoder, Router, Dense, Conv
from utils.model_utils import compute_posterior
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.training_utils import calc_aug_loss

class SmallTreeVAE(nn.Module):
    """
        A class used to represent a sub-tree VAE with one root and two children.

        SmallTreeVAE specifies a sub-tree of TreeVAE with one root and two children. It is used in the
        growing procedure of TreeVAE. At each growing step a new SmallTreeVAE is attached to a leaf of TreeVAE and
        trained separately to reduce computational time.

        Attributes
        ----------
        activation : str
            The name of the activation function for the reconstruction loss [sigmoid, mse]
        loss : models.losses
            The loss function used by the decoder to reconstruct the input
        act_function : str
            The name of the activation function used in the hidden layers of the networks
        spectral_norm : bool
            Whether to use spectral normalization
        alpha : float
            KL-annealing weight initialization
        dropout_router : float
            Dropout rate in router networks
        res_connections : bool
            Whether to use residual connection in the transformation and bottom-up layers
        depth : int
            The depth at which the sub-tree will be attached (root has depth 0 and a root with two leaves has depth 1)
        inp_shape : int
            The image resolution of the input data (if images of 32x32x3 then 32)
        inp_channel : int
            The number of input channels (if images of 32x32x3 then 3)
        latent_channel : int
            The number of latent channels used in the sub-tree
        bottom_up_channel : int
            The number of channels used in the bottom-up layers
        representation_dim : int
            The dimension of the latent representation in the bottom-up layers and the sub-tree
        augment : bool
             Whether to use contrastive learning through augmentation, if False no augmentation is used
        augmentation_method : str
            The type of augmentation method used
        aug_decisions_weight : str
            The weight of the contrastive loss used in the decisions
        denses : nn.ModuleList
            List of dense layers for the sharing of top-down and bottom-up (MLPs) associated with each of the two leaf
             node of the tree from left to right.
        transformations : nn.ModuleList
            List of transformations (MLPs) associated with each of the two leaf node of the sub-tree from left to right
        decision : Router
            The decision associated with the root of the sub-tree.
        decoders : nn.ModuleList
            List of two decoders one for each leaf of the sub-tree
        decision_q : str
            The decision of the bottom-up associated with the root of the sub-tree

        Methods
        -------
        forward(x)
            Compute the forward pass of the SmallTreeVAE model and return a dictionary of losses.
    """
    def __init__(self, depth, **kwargs):
        """
        Parameters
        ----------
        depth: int
            The depth at which the sub-tree will be attached to TreeVAE
        kwargs : dict
            A dictionary of attributes (see config file).
        """
        super(SmallTreeVAE, self).__init__()
        self.kwargs = kwargs

        # Activation function for final layer of the decoder, needed for the reconstruction loss
        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        
        # Whether to change dimensionality using Conv2d & ConvTranspose2d or Downsample & Upsample
        if 'dim_mod_conv' not in self.kwargs:
            self.dim_mod_conv = False
        else:
            self.dim_mod_conv = self.kwargs['dim_mod_conv']

        # Activation function used in the hidden layers of the networks
        self.act_function = self.kwargs['act_function']
        # Spectral normalization
        self.spectral_norm = self.kwargs['spectral_norm']
        # KL-annealing weight initialization
        self.alpha=self.kwargs['kl_start']
        # dropout rate in router networks
        self.dropout_router = self.kwargs['dropout_router']
        # Whether to use residual connection in the transformation and bottom-up layers
        self.res_connections = self.kwargs['res_connections']
        # Depth of the sub-tree
        self.depth = depth
        if -(self.depth-1) == 0:
            last_index = None
        else:
            last_index = -(self.depth-1)
        # Parameters for latent representation size and channels
        self.representation_dim = self.kwargs['representation_dim']
        latent_channels = self.kwargs['latent_channels']
        bottom_up_channels = self.kwargs['bottom_up_channels']
        latent_channels_gen = latent_channels[-(self.depth+1):last_index] # e.g. latent_channel_sizes = 32, 16, depth 2
        self.latent_channel = latent_channels_gen[::-1]
        bottom_up_channels_gen = bottom_up_channels[-(self.depth+1):last_index]
        self.bottom_up_channel = bottom_up_channels_gen[::-1]
        # Input shape and channels
        self.inp_shape = self.kwargs['inp_shape']
        self.inp_channel = self.kwargs['inp_channels']
        # Augmentation parameters for contrastive learning
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']

        # Define the networks for the sub-tree
        # -> 2 children and 1 root nodes
        # -> 2 decoders, 2 transformations, 2 denses, 1 decision, 1 decision_q
        self.denses = nn.ModuleList([Dense(self.bottom_up_channel[1], self.latent_channel[1],
                                           self.spectral_norm) for _ in range(2)])
        self.transformations = nn.ModuleList([Conv(input_channels=self.latent_channel[0],
                                                   output_channels=self.bottom_up_channel[1],
                                                   encoded_channels=self.latent_channel[1],
                                                   res_connections=self.res_connections,
                                                   act_function=self.act_function,
                                                   spectral_normalization=False) for _ in range(2)])
        self.decision = Router(input_channels=self.latent_channel[0],
                               rep_dim=self.representation_dim,
                               hidden_units=self.bottom_up_channel[0],
                               dropout=self.dropout_router,
                               act_function=self.act_function,
                               spectral_normalization=False)
        self.decision_q = Router(input_channels=self.bottom_up_channel[0],
                                 rep_dim=self.representation_dim,
                                 hidden_units=self.bottom_up_channel[0],
                                 dropout=self.dropout_router,
                                 act_function=self.act_function,
                                 spectral_normalization=False)
        self.decoders = nn.ModuleList([get_decoder(architecture=self.kwargs['encoder'],
                                                   input_shape=self.representation_dim,
                                                   input_channels=self.latent_channel[1],
                                                   output_shape=self.inp_shape,
                                                   output_channels=self.inp_channel,
                                                   activation=self.activation,
                                                   act_function=self.act_function,
                                                   spectral_normalization=False, 
                                                   dim_mod_conv=self.dim_mod_conv) for _ in range(2)])

    def forward(self, x, z_parent, p, bottom_up):
        """
        Forward pass of the SmallTreeVAE model.

        Parameters
        ----------
        x : tensor
            Input data (batch-size, input-size)
        z_parent: tensor
            The embeddings of the parent of the two children of SmallTreeVAE (which are the embeddings of the TreeVAE
            leaf where the SmallTreeVAE will be attached)
        p: list
            Probabilities of falling into the selected TreeVAE leaf where the SmallTreeVAE will be attached
        bottom_up: list
            The list of bottom-up transformations [encoder, MLP, MLP, ...] up to the root

        Returns
        -------
        dict
            a dictionary
            {'rec_loss': reconstruction loss,
            'kl_decisions': the KL loss of the decisions,
            'kl_nodes': the KL loss of the nodes,
            'aug_decisions': the weighted contrastive loss,
            'p_c_z': the probability of each sample to be assigned to each leaf with size: #samples x #leaves,
            }
        """
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

        kl_nodes_loss = torch.clamp(kl_nodes, min=-10, max=1e10)

        # Probability of falling in each leaf
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)    

        return {
            'rec_loss': rec_loss,
            'kl_decisions': kl_decisions,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
        }