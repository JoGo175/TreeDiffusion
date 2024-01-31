"""
TreeVAE model.
"""
import torch
import torch.nn as nn
import torch.distributions as td
from utils.model_utils import construct_tree, compute_posterior
from models.networks import get_encoder, get_decoder, MLP, Router, Dense, Conv, contrastive_projection
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.model_utils import return_list_tree
from utils.training_utils import calc_aug_loss

import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import display_image

class TreeVAE(nn.Module):
    def __init__(self, **kwargs):
        super(TreeVAE, self).__init__()
        self.kwargs = kwargs
        
        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        # KL-annealing weight initialization
        self.alpha=torch.tensor(self.kwargs['kl_start'])

        # dropout in router
        self.dropout_router = self.kwargs['dropout_router']
        # skip connection in the MLPs of the transformations
        self.skip_con_transformation = self.kwargs['skip_con_transformation']
        # saving important variables to initialize the tree
        self.latent_channels = self.kwargs['latent_channels']
        self.bottom_up_channels = self.kwargs['bottom_up_channels']
        self.representation_dim = self.kwargs['representation_dim']

        # check that the number of layers for bottom up is equal to top down
        if len(self.latent_channels) != len(self.bottom_up_channels):
            raise ValueError('Model is mispecified!!')
        self.depth = self.kwargs['initial_depth']
        self.inp_shape = self.kwargs['inp_shape']
        self.inp_channel = self.kwargs['inp_channels']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']
        self.return_x = torch.tensor([False])
        self.return_bottomup = torch.tensor([False])
        self.return_elbo = torch.tensor([False])

        # bottom up: the inference chain that from x computes the d units till the root
        if self.activation == "mse":
            encoder = get_encoder(architecture=self.kwargs['encoder'], input_shape=int((self.inp_shape)**0.5),
                                  input_channels=self.inp_channel, output_shape=self.representation_dim,
                                  output_channels=self.bottom_up_channels[0])
        else:
            encoder = get_encoder(architecture=self.kwargs['encoder'], input_shape=int((self.inp_shape)**0.5),
                                  input_channels=self.inp_channel, output_shape=self.representation_dim,
                                  output_channels=self.bottom_up_channels[0])

        self.bottom_up = nn.ModuleList([encoder])
        for i in range(1, len(self.bottom_up_channels)):
            self.bottom_up.append(Conv(self.bottom_up_channels[i-1], self.latent_channels[i], skip_connection=self.skip_con_transformation))

        # MLP's if we use contrastive loss on d's
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']]) > 0:
            self.contrastive_mlp = nn.ModuleList([])
            for i in range(0, len(self.bottom_up_channels)):
                self.contrastive_mlp.append(contrastive_projection(input_size=self.bottom_up_channels[i],
                                                                   rep_dim=self.representation_dim))

        # top down: the generative model that from x computes the prior prob of all nodes from root till leaves
        # it has a tree structure which is constructed by passing a list of transformations and routers from root to
        # leaves visiting nodes layer-wise from left to right
        # N.B. root has None as transformation and leaves have None as routers
        # the encoded sizes and layers are reversed from bottom up
        # e.g. for bottom up [MLP(256, 32), MLP(128, 16), MLP(64, 8)] the list of top-down transformations are
        # [None, MLP(16, 64), MLP(16, 64), MLP(32, 128), MLP(32, 128), MLP(32, 128), MLP(32, 128)]

        # select the top down generative networks
        encoded_size_gen = self.latent_channels[-(self.depth+1):]  # e.g. encoded_sizes 32,16,8, depth 1
        encoded_size_gen = encoded_size_gen[::-1]  # encoded_size_gen = 16,8 => 8,16
        layers_gen = self.bottom_up_channels[-(self.depth+1):]  # e.g. encoded_sizes 256,128,64, depth 1
        layers_gen = layers_gen[::-1]  # encoded_size_gen = 128,64 => 64,128

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = nn.ModuleList([None])
        self.denses = nn.ModuleList([Dense(layers_gen[0], encoded_size_gen[0])])  
        for i in range(self.depth):
            for j in range(2 ** (i + 1)):
                self.transformations.append(Conv(encoded_size_gen[i], encoded_size_gen[i+1], skip_connection=self.skip_con_transformation))
                self.denses.append(Dense(layers_gen[i+1], encoded_size_gen[i+1])) # Dense at depth i+1 from bottom-up to top-down

        self.decisions = nn.ModuleList([])
        for i in range(self.depth):
            for j in range(2 ** i):
                self.decisions.append(Router(encoded_size_gen[i], rep_dim=self.representation_dim,
                                             hidden_units=layers_gen[i], dropout=self.dropout_router)) # Router at node of depth i

        # decoders = [None, None, None, Dec, Dec, Dec, Dec]
        self.decoders = nn.ModuleList([None for i in range(self.depth) for j in range(2 ** i)])
        # the leaves do not have decisions but have decoders
        for _ in range(2 ** (self.depth)):
            self.decisions.append(None)
            self.decoders.append(get_decoder(architecture=self.kwargs['encoder'], input_shape=self.representation_dim,
                                             input_channels=self.latent_channels[-1], output_shape=int((self.inp_shape)**0.5),
                                             output_channels= self.inp_channel, activation=self.activation))

        # bottom-up decisions
        self.decisions_q = nn.ModuleList([])
        for i in range(self.depth):
            for _ in range(2 ** i):
                self.decisions_q.append(Router(layers_gen[i], rep_dim=self.representation_dim,
                                               hidden_units=layers_gen[i], dropout=self.dropout_router))
        for _ in range(2 ** (self.depth)):
            self.decisions_q.append(None)

        # construct the tree
        self.tree = construct_tree(transformations=self.transformations, routers=self.decisions,
                                        routers_q=self.decisions_q, denses=self.denses, decoders=self.decoders)

    def forward(self, x):
        # Small constant to prevent numerical instability
        epsilon = 1e-7  
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []
        emb_contr = []

        for i in range(0, len(self.bottom_up_channels)):
            d, _, _ = self.bottom_up[i](d)

            # Pass through contrastive MLP's
            if 'instancewise_full' in self.augmentation_method:
                _, emb_c, _ = self.contrastive_mlp[i](d)
                emb_contr.append(emb_c)
            elif 'instancewise_first' in self.augmentation_method:
                if i == 0:
                    _, emb_c, _ = self.contrastive_mlp[i](d)
                    emb_contr.append(emb_c)

            # store bottom-up embeddings for top-down
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]
        # initializate KL losses
        kl_nodes_tot = torch.zeros(len(x), device=device)
        kl_decisions_tot = torch.zeros(len(x), device=device)
        aug_decisions_loss = torch.zeros(1, device=device)
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        decoder_magnitudes = []
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:  
                # here we are in the root
                # standard gaussian
                z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat, device=device), torch.ones_like(z_sigma_q_hat, device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                # sampled z is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled paren
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                # z_p is 3-dimensional (channel, height, width)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 3)
            z_sample = z.rsample()

            # compute KL node between z, z_p, both are distributions for (sample, channel, height, width)
            # need multivariate gaussian KL
            kl_node = prob * td.kl_divergence(z, z_p)
            kl_node = torch.clamp(kl_node, min=-1, max=1000)
        
            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            if node.router is not None:
                # we are in the internal nodes (not leaves)
                prob_child_left = node.router(z_sample).squeeze()
                prob_child_left_q = node.routers_q(d).squeeze()

                kl_decisions = prob_child_left_q * (epsilon + prob_child_left_q / (prob_child_left + epsilon)).log() + \
                                (1 - prob_child_left_q) * (epsilon + (1 - prob_child_left_q) / (1 - prob_child_left + epsilon)).log()
                
                if self.training is True and self.augment is True and 'simple' not in self.augmentation_method:
                    if depth_level == 0:
                        # Only do contrastive loss on representations once
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=emb_contr)
                    else:
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=[])

                kl_decisions = prob * kl_decisions
                kl_decisions_tot += kl_decisions

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})
                # get decoder weights
                params = []
                for param in dec.parameters():
                    if param.requires_grad:
                        params.append(param)
                params = torch.cat([param.view(-1) for param in params])
                decoder_magnitudes.append(torch.norm(params, p=2))


            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        kl_nodes_loss = torch.clamp(torch.mean(kl_nodes_tot), min=-10, max=1e10)
        kl_decisions_loss = torch.mean(kl_decisions_tot)
        kl_root_loss = torch.mean(kl_root)

        # p_c_z is the probability of reaching a leaf and should be of shape [batch_size, num_clusters]
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        
        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)

        return_dict = {
            'rec_loss': rec_loss,
            'weights': leaves_prob,
            'kl_root': kl_root_loss,
            'kl_decisions': kl_decisions_loss,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
            'node_leaves': node_leaves,
            'decoder_magnitudes': decoder_magnitudes
        }

        if self.return_elbo:
            return_dict['elbo_samples'] = kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses

        if self.return_bottomup: 
            return_dict['bottom_up'] = encoders

        if self.return_x:
            return_dict['input'] = x

        return return_dict


    def compute_leaves(self):
        # returns leaves of the tree
        list_nodes = [{'node': self.tree, 'depth': 0}]
        nodes_leaves = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level = current_node['node'], current_node['depth']
            if node.router is not None:
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1})
            elif node.router is None and node.decoder is None:
                # we are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append({'node': child, 'depth': depth_level + 1})                
            else:
                nodes_leaves.append(current_node)
        return nodes_leaves


    def compute_depth(self):
        # computes depth of the tree
        nodes_leaves = self.compute_leaves()
        d = []
        for i in range(len(nodes_leaves)):
            d.append(nodes_leaves[i]['depth'])
        return max(d)

    def attach_smalltree(self, node, small_model):
        # attaching a (trained) smalltree to the full tree
        assert node.left is None and node.right is None
        node.router = small_model.decision
        node.routers_q = small_model.decision_q
        node.decoder = None
        for j in range(2):
            dense = small_model.denses[j]
            transformation = small_model.transformations[j]
            decoder = small_model.decoders[j]
            node.insert(transformation, None, None, dense, decoder)
        
        transformations, routers, denses, decoders, routers_q = return_list_tree(self.tree)
        
        self.decisions_q = routers_q
        self.transformations = transformations
        self.decisions = routers
        self.denses = denses
        self.decoders = decoders
        self.depth = self.compute_depth()
        return


    def compute_reconstruction(self, x):
        assert self.training is False
        epsilon = 1e-7
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.bottom_up_channels)):
            d, _, _ = self.bottom_up[i](d)
            # store the bottom-up layers for the top down computation
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]

        # initializate KL losses
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        while len(list_nodes) != 0:

            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # if we are in the internal nodes (not leaves)
            if node.router is not None:

                prob_child_left_q = node.routers_q(d).squeeze()

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        return reconstructions, node_leaves

    def generate_images(self, n_samples, device):
        assert self.training is False
        epsilon = 1e-7
        sizes = self.latent_channels
        rep_dim = self.representation_dim
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(n_samples, device=device), 'z_parent_sample': None}]
        leaves_prob = []
        reconstructions = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            if depth_level == 0:
                z_mu_p, z_sigma_p = torch.zeros([n_samples, sizes[-1], rep_dim, rep_dim], device=device), torch.ones([n_samples, sizes[-1], rep_dim, rep_dim], device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p)), 3)
                z_sample = z_p.rsample()

            else:
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                z_sample = z_p.rsample()

            if node.router is not None:
                prob_child_left = node.router(z_sample).squeeze()
                prob_node_left, prob_node_right = prob * prob_child_left, prob * (
                        1 - prob_child_left)
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})

            elif node.decoder is not None:
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        
        return reconstructions, p_c_z

    def posterior_parameters(self) :

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0}]
        posterior_mu = []
        posterior_sigma = []

        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level = current_node['node'], current_node['depth']
            # access deterministic bottom up mu and sigma hat (computed above)
            if depth_level == 0:
                posterior_mu.append(torch.zeros_like(node.dense.mu))
                posterior_sigma.append(torch.ones_like(node.dense.sigma))
            else:
                posterior_mu.append(node.dense.mu)
                posterior_sigma.append(node.dense.sigma)

            if node.router is not None:
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1})
            
            elif node.router is None and node.decoder is None:
                # we are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append({'node': child, 'depth': depth_level + 1})

        return posterior_mu, posterior_sigma

    def get_node_embeddings(self, x):
        assert self.training == False  # Assuming self refers to an instance of your class
        epsilon = 1e-7
        device = x.device

        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.bottom_up_channels)):
            d, _, _ = self.bottom_up[i](d)
            # store the bottom-up layers for the top down computation
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]

        # Create a list to store node information
        node_info_list = []

        while len(list_nodes) != 0:
            # Store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            # Access deterministic bottom-up mu and sigma hat (computed above)
            d = encoders[-(1 + depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # The generative mu and sigma are the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # Compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # Store information in the list
            node_info = {'prob': prob, 'z_sample': z_sample}
            node_info_list.append(node_info)

            if node.router is not None:
                prob_child_left_q = node.routers_q(d).squeeze()

                # We are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})

            elif node.decoder is None and (node.left is not None or node.right is not None):
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node_right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        return node_info_list
    
    def get_latent_z(self, x):
        # TODO: fix this function
        # get the latent z from the node embeddings with the highest probability

        node_info_list = self.get_node_embeddings(x)

        # Initialize a tensor to keep track of the indices of the highest probability nodes
        max_prob_indices = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        max_probs = torch.zeros(x.size(0), device=x.device)

        # Iterate over each node's info and update the indices based on the highest probability
        for i, node_info in enumerate(node_info_list):
            prob = node_info['prob']
            mask = prob > max_probs
            max_probs[mask] = prob[mask]
            max_prob_indices[mask] = i

        # Use the indices to select the corresponding z_samples
        max_z_samples = torch.stack([node_info_list[idx]['z_sample'] for idx in max_prob_indices])

        return max_z_samples
    
    def forward_recons(self, x, max_leaf):
        res = self.compute_reconstruction(x)

        max_z_sample = []
        max_recon = []
        leaf_ind = []

        nodes = res[1]
        for i in range(len(nodes[0]['prob'])):
            probs = [node['prob'][i] for node in nodes]
            z_sample = [node['z_sample'][i] for node in nodes]
            if max_leaf:
                ind = probs.index(max(probs))
            else:
                ind = torch.multinomial(torch.stack(probs), 1).item()

            max_z_sample.append(z_sample[ind])
            max_recon.append(res[0][ind][i])
            leaf_ind.append(ind)

        # z = torch.stack(max_z_sample)
        z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(x.device)
        cond = torch.stack(max_recon)

        return cond, z









