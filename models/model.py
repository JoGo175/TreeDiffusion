"""
TreeVAE model.
"""
import torch
import torch.nn as nn
import torch.distributions as td
from utils.model_utils import construct_tree, compute_posterior
from models.networks import get_encoder, get_decoder, Router, Dense, Conv, contrastive_projection
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.model_utils import return_list_tree
from utils.training_utils import calc_aug_loss

class TreeVAE(nn.Module):
    """
        A class used to represent a tree-based VAE.

        TreeVAE specifies a Variational Autoencoder with a tree structure posterior distribution of latent variables.
        It is defined by a bottom-up chain of deterministic transformations that from the input x compute the root
        representation of the data, and a probabilistic top-down architecture which takes the form of a tree. The
        top down tree is described by the probability distribution of its node (which depends on their parents) and the
        probability distribution of the decisions (what is the probability of following a certain path in the tree).
        Each node of the tree is described by the class Node in utils.model_utils.

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
        latent_channels : list
            A list of latent channels for each depth of the tree from the bottom to the root
        bottom_up_channels : list
            A list of bottom-up channels for each depth of the tree from the bottom to the root
        representation_dim : int
            The dimension of the latent representation after the encoder and before the decoder
        depth : int
            The depth of the tree (root has depth 0 and a root with two leaves has depth 1)
        inp_shape : int
            The image resolution of the input data (if images of 32x32x3 then 32)
        inp_channel : int
            The number of input channels (if images of 32x32x3 then 3)
        augment : bool
             Whether to use contrastive learning through augmentation, if False no augmentation is used
        augmentation_method : str
            The type of augmentation method used
        aug_decisions_weight : str
            The weight of the contrastive loss used in the decisions
        return_x : float
            Whether to return the input in the return dictionary of the forward method
        return_elbo : float
            Whether to return the sample-specific elbo in the return dictionary of the forward method
        return_bottomup : float
            Whether to return the list of bottom-up transformations (including encoder)
        bottom_up : str
            The list of bottom-up transformations [encoder, MLP, MLP, ...] up to the root
        contrastive_mlp : list
            The list of transformations from the bottom-up embeddings to the latent spaces
            in which the contrastive losses are applied
        transformations : list
            List of transformations (MLPs) associated with each node of the tree from root to bottom (left to right)
        denses : list
            List of dense layers for the sharing of top-down and bottom-up (MLPs) associated with each node of the tree
            from root to bottom (left to right).
        decisions : list
            List of decisions associated with each node of the tree from root to bottom (left to right)
        decoders : list
            List of decoders one for each leaf
        decisions_q : str
            List of decisions of the bottom-up associated with each node of the tree from root to bottom (left to right)
        tree : utils.model_utils.Node
            The root node of the tree

        Methods
        -------
        forward(x)
            Compute the forward pass of the treeVAE model and return a dictionary of losses and optional outputs
            (like input, bottom-up and sample-specific elbo) when needed.
        compute_leaves()
            Return a list of leaf-nodes from left to right of the current tree (self.tree).
        compute_depth()
            Calculate the depth of the given tree (self.tree).
        attach_smalltree(node, small_model)
            Attach a sub tree (small_model) to the given node of the current tree.
        compute_reconstruction(x)
            Given the input x, it computes the reconstructions.
        generate_images(n_samples, device)
            Generate n_samples new images by sampling from the root and propagating through the entire tree.
        """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            A dictionary of attributes (see config file).
        """
        super(TreeVAE, self).__init__()
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
        self.alpha = torch.tensor(self.kwargs['kl_start'])
        # Dropout rate in router networks
        self.dropout_router = self.kwargs['dropout_router']
        # Whether to use residual connection in the transformation and bottom-up layers
        self.res_connections = self.kwargs['res_connections']
        # Parameters for latent representation size and channels
        self.latent_channels = self.kwargs['latent_channels']
        self.bottom_up_channels = self.kwargs['bottom_up_channels']
        self.representation_dim = self.kwargs['representation_dim']
        # check that the number of layers for bottom up is equal to top down
        if len(self.latent_channels) != len(self.bottom_up_channels):
            raise ValueError('Model is mispecified!!')
        # initial depth of the tree
        self.depth = self.kwargs['initial_depth']
        # Input shape and channels
        self.inp_shape = self.kwargs['inp_shape']
        self.inp_channel = self.kwargs['inp_channels']
        # Augmentation parameters for contrastive learning
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']
        # Return parameters
        self.return_x = torch.tensor([False])
        self.return_bottomup = torch.tensor([False])
        self.return_elbo = torch.tensor([False])

        # Construct the model architecture and the tree structure

        # bottom-up: the inference chain that from x computes the d units till the root
        encoder = get_encoder(architecture=self.kwargs['encoder'], input_shape=self.inp_shape,
                              input_channels=self.inp_channel, output_shape=self.representation_dim,
                              output_channels=self.bottom_up_channels[0], act_function=self.act_function,
                              spectral_normalization=self.spectral_norm, dim_mod_conv=self.dim_mod_conv)

        self.bottom_up = nn.ModuleList([encoder])
        for i in range(1, len(self.bottom_up_channels)):
            self.bottom_up.append(Conv(input_channels=self.bottom_up_channels[i-1],
                                       output_channels=self.bottom_up_channels[i],
                                       encoded_channels=self.latent_channels[i],
                                       res_connections=self.res_connections, act_function=self.act_function,
                                       spectral_normalization=self.spectral_norm))

        # contrastive projection networks if we use contrastive loss on d's
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']]) > 0:
            self.contrastive_mlp = nn.ModuleList([])
            for i in range(0, len(self.bottom_up_channels)):
                self.contrastive_mlp.append(contrastive_projection(input_size=self.bottom_up_channels[i],
                                                                   rep_dim=self.representation_dim,
                                                                   act_function=self.act_function))

        # top-down: the generative model that from x computes the prior prob of all nodes from root till leaves,
        # it has a tree structure which is constructed by passing a list of transformations and routers from root to
        # leaves visiting nodes layer-wise from left to right.
        # N.B. root has None as transformation and leaves have None as routers
        # the encoded channel sizes and layers are reversed from bottom up
        # e.g. for bottom up [MLP(256, 32), MLP(128, 16), MLP(64, 8)] the list of top-down transformations are
        # [None, MLP(16, 64), MLP(16, 64), MLP(32, 128), MLP(32, 128), MLP(32, 128), MLP(32, 128)]

        # select the top-down generative networks
        encoded_size_gen = self.latent_channels[-(self.depth+1):]  # e.g. encoded_sizes 32,16,8, depth 1
        encoded_size_gen = encoded_size_gen[::-1]  # encoded_size_gen = 16,8 => 8,16
        layers_gen = self.bottom_up_channels[-(self.depth+1):]  # e.g. encoded_sizes 256,128,64, depth 1
        layers_gen = layers_gen[::-1]  # encoded_size_gen = 128,64 => 64,128

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = nn.ModuleList([None])
        self.denses = nn.ModuleList([Dense(layers_gen[0], encoded_size_gen[0])])
        # attach the rest of transformations and dense layers for each node
        for i in range(self.depth):
            for j in range(2 ** (i + 1)):
                # Transformation Conv from depth i to i+1
                self.transformations.append(Conv(input_channels=encoded_size_gen[i],
                                                 output_channels=layers_gen[i+1],
                                                 encoded_channels=encoded_size_gen[i+1],
                                                 res_connections=self.res_connections, act_function=self.act_function,
                                                 spectral_normalization=False))
                # Dense at depth i+1 from bottom-up to top-down
                self.denses.append(Dense(input_channels=layers_gen[i+1], output_channels=encoded_size_gen[i+1],
                                         spectral_normalization=False))

        # compute the list of decisions for both bottom-up (decisions_q) and top-down (decisions)
        # for each node of the tree
        self.decisions = nn.ModuleList([])
        self.decisions_q = nn.ModuleList([])
        for i in range(self.depth):
            for _ in range(2 ** i):
                # Router at node of depth i
                self.decisions.append(Router(input_channels=encoded_size_gen[i], rep_dim=self.representation_dim,
                                             hidden_units=layers_gen[i], dropout=self.dropout_router,
                                             act_function=self.act_function,
                                             spectral_normalization=False))

                self.decisions_q.append(Router(input_channels=layers_gen[i], rep_dim=self.representation_dim,
                                               hidden_units=layers_gen[i], dropout=self.dropout_router,
                                               act_function=self.act_function,
                                               spectral_normalization=False))
        # # the leaves do not have decisions (we set it to None)
        for _ in range(2 ** (self.depth)):
            self.decisions.append(None)
            self.decisions_q.append(None)

        # compute the list of decoders to attach to each node, note that internal nodes do not have a decoder
        # e.g. for a tree with depth 2: decoders = [None, None, None, Dec, Dec, Dec, Dec]
        self.decoders = nn.ModuleList([None for i in range(self.depth) for j in range(2 ** i)])
        for _ in range(2 ** (self.depth)):
            self.decoders.append(get_decoder(architecture=self.kwargs['encoder'], input_shape=self.representation_dim,
                                             input_channels=encoded_size_gen[i+1], output_shape=self.inp_shape,
                                             output_channels= self.inp_channel, activation=self.activation,
                                             act_function=self.act_function, spectral_normalization=False, 
                                             dim_mod_conv=self.dim_mod_conv))

        # construct the tree
        self.tree = construct_tree(transformations=self.transformations, routers=self.decisions,
                                   routers_q=self.decisions_q, denses=self.denses, decoders=self.decoders)

    def forward(self, x):
        """
        Forward pass of the treeVAE model.

        Parameters
        ----------
        x : tensor
            Input data (batch-size, input-size)

        Returns
        -------
        dict
            a dictionary
            {'rec_loss': reconstruction loss,
            'kl_root': the KL loss of the root,
            'kl_decisions': the KL loss of the decisions,
            'kl_nodes': the KL loss of the nodes,
            'aug_decisions': the weighted contrastive loss,
            'p_c_z': the probability of each sample to be assigned to each leaf with size: #samples x #leaves,
            'node_leaves': a list of leaf nodes, each one described by a dictionary
                            {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
            }
        """
        # Small constant to prevent numerical instability
        epsilon = 1e-7  
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []
        emb_contr = []

        for i in range(0, len(self.bottom_up_channels)):
            d, _, _ = self.bottom_up[i](d)
            # store bottom-up embeddings for top-down
            encoders.append(d)

            # Pass through contrastive projection networks if contrastive learning is selected
            if 'instancewise_full' in self.augmentation_method:
                _, emb_c, _ = self.contrastive_mlp[i](d)
                emb_contr.append(emb_c)
            elif 'instancewise_first' in self.augmentation_method:
                if i == 0:
                    _, emb_c, _ = self.contrastive_mlp[i](d)
                    emb_contr.append(emb_c)

        # create a list of nodes of the tree that need to be processed, self.tree is the root of the tree
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None, 'id': 0, 'parent': None}]
        # list of nodes to be outputted, should include 'node_id', 'depth', 'prob', 'z_sample', 'parent_id'
        node_info_return = []
        id_ind = 0
        leaf_ind = 0

        # initializate KL losses
        kl_nodes_tot = torch.zeros(len(x), device=device)
        kl_decisions_tot = torch.zeros(len(x), device=device)
        aug_decisions_loss = torch.zeros(1, device=device)
        leaves_prob = []
        reconstructions = []
        node_leaves = []

        # iterates over all nodes in the tree
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            # here we are in the root
            if depth_level == 0:
                # the root has a standard gaussian prior
                z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat, device=device), torch.ones_like(z_sigma_q_hat, device=device)
                # z_p is 3-dimensional (channel, height, width)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                # the samples z (from q(z|x)) is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat

            # otherwise we are in the rest of the nodes of the tree
            else:
                # the generative probability distribution of internal nodes is a gaussian with mu and sigma that are
                # the outputs of the top-down network conditioned on the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                # z_p is 3-dimensional (channel, height, width)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 3)
            z_sample = z.rsample()

            # store the node information for output
            leaf_info = None
            if node.router is None and node.decoder is not None:
                leaf_info = leaf_ind
                leaf_ind += 1
            node_info_return.append({'node_id': current_node['id'], 'depth': depth_level, 'prob': prob,
                                     'z_sample': z_sample, 'parent_id': current_node['parent'], 'leaf_id': leaf_info})

            # compute KL node between z, z_p, both are distributions for (channel, height, width)
            # need multivariate gaussian KL
            kl_node = prob * td.kl_divergence(z, z_p)
            kl_node = torch.clamp(kl_node, min=-1, max=1000)
        
            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            # if there is a router (i.e. decision probability) then we are in the internal nodes (not leaves)
            if node.router is not None:
                # compute the probability of the sample to go to the left child
                prob_child_left = node.router(z_sample).squeeze()
                prob_child_left_q = node.routers_q(d).squeeze()

                # compute the KL of the decisions
                kl_decisions = prob_child_left_q * (epsilon + prob_child_left_q / (prob_child_left + epsilon)).log() + \
                                (1 - prob_child_left_q) * (epsilon + (1 - prob_child_left_q) / (1 - prob_child_left + epsilon)).log()
                kl_decisions = prob * kl_decisions
                kl_decisions_tot += kl_decisions

                # compute the contrastive loss of the embeddings and the decisions
                if self.training is True and self.augment is True and 'simple' not in self.augmentation_method:
                    if depth_level == 0:
                        # compute the contrastive loss for all the bottom-up representations
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=emb_contr)
                    else:
                        # compute the contrastive loss for the decisions
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=[])

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)
                node_left, node_right = node.left, node.right
                id_ind += 1
                list_nodes.append({'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left,
                                   'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})
                id_ind += 1
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})

            # if there is a decoder then we are in one of the leaf nodes
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})

            # here we are in an internal node with pruned leaves and thus only have one child
            elif node.router is None and node.decoder is None:
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                id_ind += 1
                list_nodes.append({'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample,
                                'id': id_ind, 'parent': current_node['id']})

        kl_nodes_loss = torch.clamp(torch.mean(kl_nodes_tot), min=-10, max=1e10)
        kl_decisions_loss = torch.mean(kl_decisions_tot)
        kl_root_loss = torch.mean(kl_root)

        # p_c_z is the probability of reaching a leaf and is of shape [batch_size, num_clusters]
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        
        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)

        return_dict = {
            'rec_loss': rec_loss,
            'kl_root': kl_root_loss,
            'kl_decisions': kl_decisions_loss,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
            'node_leaves': node_leaves,
            'reconstructions': reconstructions,
            'node_info': node_info_return
        }

        if self.return_elbo:
            return_dict['elbo_samples'] = kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses

        if self.return_bottomup: 
            return_dict['bottom_up'] = encoders

        if self.return_x:
            return_dict['input'] = x

        return return_dict


    def compute_leaves(self):
        """
        Computes the leaves of the tree

        Returns
        -------
        list
            A list of the leaves from left to right.
            A leaf is defined by a dictionary: {'node': leaf node, 'depth': depth of the leaf node}.
            A leaf node is defined by the class Node in utils.model_utils.
        """
        # iterate over all nodes in the tree to find the leaves
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
        """
        Computes the depth of the tree

        Returns
        -------
        int
            The depth of the tree (the root has depth 0 and a root with two leaves had depth 1).
        """
        # computes depth of the tree
        nodes_leaves = self.compute_leaves()
        d = []
        for i in range(len(nodes_leaves)):
            d.append(nodes_leaves[i]['depth'])
        return max(d)

    def attach_smalltree(self, node, small_model):
        """
        Attach a trained small tree of the class SmallTreeVAE (models.model_smalltree) to the given node of the full
        TreeVAE. The small tree has one root and two leaves. It does not return anything but changes self.tree

        Parameters
        ----------
        node : utils.model_utils.Node
            The selected node of TreeVAE where to attach the sub-tree, which was trained separately.
        small_model: models.model_smalltree.SmallTreeVAE
            The sub-tree with one root and two leaves that needs to be attached to TreeVAE.
        """
        assert node.left is None and node.right is None
        node.router = small_model.decision
        node.routers_q = small_model.decision_q
        node.decoder = None
        for j in range(2):
            dense = small_model.denses[j]
            transformation = small_model.transformations[j]
            decoder = small_model.decoders[j]
            # insert each leaf of the small tree as child of the node of TreeVAE
            node.insert(transformation, None, None, dense, decoder)

        # once the small tree is attached we re-compute the list of transformations, routers etc
        transformations, routers, denses, decoders, routers_q = return_list_tree(self.tree)

        # we then need to re-initialize the parameters of TreeVAE
        self.decisions_q = routers_q
        self.transformations = transformations
        self.decisions = routers
        self.denses = denses
        self.decoders = decoders
        self.depth = self.compute_depth()
        return


    def compute_reconstruction(self, x):
        """
        Given the input x, it computes the reconstructions.

        Parameters
        ----------
        x: Tensor
            Input data.

         Returns
        -------
        Tensor
            The reconstructions of the input data by computing a forward pass of the model.
        List
            A list of leaf nodes, each one described by a dictionary
            {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
        """
        assert self.training is False
        epsilon = 1e-7
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.bottom_up_channels)):
            d, _, _ = self.bottom_up[i](d)
            # store the bottom-up layers for the top-down computation
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None, 'id': 0, 'parent': None}]
        # list of nodes to be outputted, should include 'node_id', 'depth', 'prob', 'z_sample', 'parent_id'
        node_info_return = []
        id_ind = 0
        leaf_ind = 0

        # initialize KL losses
        leaves_prob = []
        reconstructions = []
        node_leaves = []

        # iterate over the nodes
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
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 3)
            z_sample = z.rsample()

            # store the node information for output
            leaf_info = None
            if node.router is None and node.decoder is not None:
                leaf_info = leaf_ind
                leaf_ind += 1
            node_info_return.append({'node_id': current_node['id'], 'depth': depth_level, 'prob': prob,
                                     'z_sample': z_sample, 'parent_id': current_node['parent'], 'leaf_id': leaf_info})

            # if we are in the internal nodes (not leaves)
            if node.router is not None:

                prob_child_left_q = node.routers_q(d).squeeze()

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                id_ind += 1
                list_nodes.append({'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left,
                                   'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})
                id_ind += 1
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                   'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})

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
                id_ind += 1
                list_nodes.append({'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample,
                                   'id': id_ind, 'parent': current_node['id']})

        return reconstructions, node_leaves, node_info_return

    def generate_images(self, n_samples, device):
        """
        Generate K x n_samples new images by sampling from the root and propagating through the entire tree.
        For each sample the method generates K images, where K is the number of leaves.

        Parameters
        ----------
        n_samples: int
            Number of generated samples the function should output.
        device: torch.device
            Either cpu or gpu

         Returns
        -------
        list
            A list of K tensors containing the leaf-specific generations obtained by sampling from the root and
            propagating through the entire tree, where K is the number of leaves.
        Tensor
            The probability of each generated sample to be assigned to each leaf with size: #samples x #leaves,
        """
        assert self.training is False
        epsilon = 1e-7
        sizes = self.latent_channels
        rep_dim = self.representation_dim
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(n_samples, device=device), 'z_parent_sample': None}]
        leaves_prob = []
        reconstructions = []

        # iterates over all nodes in the tree
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            if depth_level == 0:
                z_mu_p, z_sigma_p = torch.zeros([n_samples, sizes[-1], rep_dim, rep_dim], device=device), \
                    torch.ones([n_samples, sizes[-1], rep_dim, rep_dim], device=device)
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
                # here we are in a leaf node and we attach the corresponding generations
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

    def posterior_parameters(self):
        """
        Return the posterior parameters for each node of the tree. The posterior parameters are the mu and sigma
        of the generative distribution for each node of the tree.

        Returns
        -------
        list
            A list of tensors containing the posterior mu for each node of the tree.
        """

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0}]
        posterior_mu = []
        posterior_sigma = []

        # iterates over all nodes in the tree
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
        """
        Compute the node embeddings for each node of the tree given the input x.

        Parameters
        ----------
        x: Tensor
            Input data.

        Returns
        -------
        list
            A list of dictionaries containing the node embeddings for each node of the tree.
            Each node is described by a dictionary:
            {'prob': sample-wise probability of reaching the node, 'z_sample': sample-wise leaf embedding}
        """
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

        # iterates over all nodes in the tree
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
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 3)
            z_sample = z.rsample()

            # Store information of the node
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

    
    def forward_recons(self, x, max_leaf, z_signal):
        """
        Forward pass of the TreeVAE model to compute the conditioning information that is used as input to the
        Diffusion model in Diffuse-TreeVAE.
        Samples the leaf with the highest probability or samples from the leaf probabilities to get the conditioning
        information.
        Currently, the conditioning information is simply the leaf index and the corresponding reconstruction.

        Parameters
        ----------
        x : Tensor
            Input data
        max_leaf: bool
            Whether to use the leaf with the highest probability or sample from the leaf probabilities.

        Returns
        -------
        Tensor
            The selected reconstructions.
        Tensor
            The selected leaf indices. Can later be altered to be the latent embeddings z.
        """
        # Compute the reconstructions and the leaf embeddings
        res = self.compute_reconstruction(x)
        recons = res[0]
        nodes = res[1]


        # initialize cond and z
        cond = None
        z = None

        # Save the chosen leaf_embeddings, reconstructions and the respective leaf indices
        max_z_sample = []
        max_recon = []
        leaf_ind = []

        # Iterate over the leaf nodes and select the leaf with the highest probability or sample given the leaf probs
        for i in range(len(nodes[0]['prob'])):
            probs = [node['prob'][i] for node in nodes]
            z_sample = [node['z_sample'][i] for node in nodes]

            if max_leaf:
                ind = probs.index(max(probs))
            else:
                ind = torch.multinomial(torch.stack(probs), 1).item()

            max_z_sample.append(z_sample[ind])
            max_recon.append(recons[ind][i])
            leaf_ind.append(ind)

        # leaf index + latent embeddings as conditioning signal
        if z_signal == "both":
            z1 = torch.stack(max_z_sample)
            z2 = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(x.device)
            z = [z1, z2]

        # latent embeddings as conditioning signal
        elif z_signal == "latent":
            z = torch.stack(max_z_sample)

        # leaf index as conditioning signal instead of latent embeddings, z should be (batch, 1)
        elif z_signal == "cluster_id":
            z = torch.tensor(leaf_ind, dtype=torch.float).unsqueeze(1).to(x.device)

        # reconstructions
        cond = torch.stack(max_recon)
        return cond, z

    def generate_images_and_embeddings(self, n_samples, device):
        """
        Generate K x n_samples new images by sampling from the root and propagating through the entire tree.
        For each sample the method generates K images, where K is the number of leaves.

        Parameters
        ----------
        n_samples: int
            Number of generated samples the function should output.
        device: torch.device
            Either cpu or gpu

         Returns
        -------
        list
            A list of K tensors containing the leaf-specific generations obtained by sampling from the root and
            propagating through the entire tree, where K is the number of leaves.
        Tensor
            The probability of each generated sample to be assigned to each leaf with size: #samples x #leaves,
        """
        assert self.training is False
        epsilon = 1e-7
        sizes = self.latent_channels
        rep_dim = self.representation_dim
        list_nodes = [
            {'node': self.tree, 'depth': 0, 'prob': torch.ones(n_samples, device=device), 'z_parent_sample': None, 'id': 0, 'parent': None}]
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        node_info_return = []
        id_ind = 0
        leaf_ind = 0

        # iterates over all nodes in the tree
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            if depth_level == 0:
                z_mu_p, z_sigma_p = torch.zeros([n_samples, sizes[-1], rep_dim, rep_dim], device=device), \
                    torch.ones([n_samples, sizes[-1], rep_dim, rep_dim], device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p)), 3)
                z_sample = z_p.rsample()

            else:
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 3)
                z_sample = z_p.rsample()

            # store the node information for output
            leaf_info = None
            if node.router is None and node.decoder is not None:
                leaf_info = leaf_ind
                leaf_ind += 1
            node_info_return.append({'node_id': current_node['id'], 'depth': depth_level, 'prob': prob,
                                     'z_sample': z_sample, 'parent_id': current_node['parent'],
                                     'leaf_id': leaf_info})

            if node.router is not None:
                prob_child_left = node.router(z_sample).squeeze()
                prob_node_left, prob_node_right = prob * prob_child_left, prob * (
                        1 - prob_child_left)
                node_left, node_right = node.left, node.right
                id_ind += 1
                list_nodes.append({'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left,
                                   'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})
                id_ind += 1
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                   'z_parent_sample': z_sample, 'id': id_ind, 'parent': current_node['id']})

            elif node.decoder is not None:
                # here we are in a leaf node and we attach the corresponding generations
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                id_ind += 1
                list_nodes.append({'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample,
                                   'id': id_ind, 'parent': current_node['id']})

        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        return reconstructions, node_leaves, p_c_z, node_info_return


