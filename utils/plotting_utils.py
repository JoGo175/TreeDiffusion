'''
Utility functions for plotting the tree graph with and without scatter plots as nodes.
'''
import numpy as np
import torch
import torch.distributions as td
from matplotlib import pyplot as plt
from utils.model_utils import compute_posterior
import re
import networkx as nx
from sklearn.decomposition import PCA
from matplotlib.patches import ConnectionPatch


def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''
    Encodes the hierarchy for the tree layout in a graph.
    Adopted from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
    If there is a cycle that is reachable from root, then this will see infinite recursion.
    Parameters
    ----------
    G: the graph
    root: the root node
    levels: a dictionary
    key: level number (starting from 0)
    value: number of nodes in this level
    width: horizontal space allocated for drawing
    height: vertical space allocated for drawing
    '''

    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """
        Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        """
        Compute the position of each node
        """
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})


def plot_tree_graph(data):
    """
    Plot the tree graph without scatter plots as nodes.
    Nodes are colored based on their type (internal or leaf).
    Internal nodes are colored lightblue and show the node_id.
    Leaf nodes are colored lightgreen and show the distribution of the labels within the cluster.
    """
    # get a '/n' before every 'tot' in each second entry of data
    data = data.copy()
    for d in data:
        if d[3] == 1:
            pattern = r'(\w+:\s\d+\.\d+|\d+:\s\d+\.\d+|\w+\s\d+|\d+\s\d+|\w+:\s\d+|\d+:\s\d+|\w+:\s\d+\s\w+|\d+:\s\d+\s\w+|\w+\s\d+\s\w+|\d+\s\d+\s\w+|\w+:\s\d+\.\d+\s\w+|\d+:\s\d+\.\d+\s\w+)'
            # Split the string using the regular expression pattern
            result = re.findall(pattern, d[1])
            # Join the resulting list to format it as desired
            d[1] = '\n'.join(result)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in data:
        node_id, label, parent_id, node_type = node
        G.add_node(node_id, label=label, node_type=node_type)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    # Get positions of graph nodes
    pos = hierarchy_pos(G, 0, levels=None, width=1, height=1)
    # Get the labels of the nodes
    labels = nx.get_node_attributes(G, 'label')
    # Initialize node color and size lists
    node_colors = []
    node_sizes = []

    # Iterate through nodes to set colors and sizes
    for node_id, node_data in G.nodes(data=True):
        if G.out_degree(node_id) == 0:  # Leaf nodes have out-degree 0
            node_colors.append('lightgreen')  
            node_sizes.append(4000)
        else: # Internal nodes
            node_colors.append('lightblue')  
            node_sizes.append(1000) 

    # Draw the graph with different node properties
    plt.figure(figsize=(10, 5))
    nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=7)
    plt.show()



# Create a function to draw scatter plots as nodes
def draw_scatter_node(node_id, node_embeddings, colors, ax, pca = True):
    """
    Draw a scatter plot for a node. The scatter plot shows the latent space of the node. 

    Parameters
    ----------
    node_id: int
        The id of the node
    node_embeddings: dict
        The node embeddings
    colors: np.array
        The colors of the observations
    ax: matplotlib axes
        The axes to draw the scatter plot on, important for the layout of the tree graph
    pca: bool
        Whether to use PCA to reduce the dimensionality of the latent space for visualization
    """

    # if list is empty --> node has been pruned
    if node_embeddings[node_id]['z_sample'] == []:
        # return empty plot
        ax.set_title(f"Node {node_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # get the latent space embeddings and the probabilities of the observations
    z_sample = node_embeddings[node_id]['z_sample'].reshape(node_embeddings[node_id]['z_sample'].shape[0], -1)
    weights = node_embeddings[node_id]['prob']

    # if pca is True, reduce the dimensionality of the latent space to 2 dimensions
    # otherwise, use the first two dimensions of the latent space
    if pca:
        pca_fit = PCA(n_components=2)
        z_sample = pca_fit.fit_transform(z_sample)

    # plot the scatter plot at the given axes
    ax.scatter(z_sample[:, 0], z_sample[:, 1], c=colors, cmap='tab10', alpha=weights, s = 0.25)
    ax.set_title(f"Node {node_id}")
    ax.set_xticks([])
    ax.set_yticks([])


# Create a function to draw scatter plots as nodes
def draw_flattened_dist_node(node_id, node_embeddings, colors, ax, pca = True):
    """
    Draw a distribution plot for a node. The distribution plot shows the first principal component of the latent space of the node.

    Parameters
    ----------
    node_id: int
        The id of the node
    node_embeddings: dict
        The node embeddings
    colors: np.array
        The colors of the observations
    ax: matplotlib axes
        The axes to draw the scatter plot on, important for the layout of the tree graph
    pca: bool
        Whether to use PCA to reduce the dimensionality of the latent space for visualization
    """

    # if list is empty --> node has been pruned
    if node_embeddings[node_id]['z_sample'] == []:
        # return empty plot
        ax.set_title(f"Node {node_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # get the latent space embeddings and the probabilities of the observations
    z_sample = node_embeddings[node_id]['z_sample'].reshape(node_embeddings[node_id]['z_sample'].shape[0], -1)
    weights = node_embeddings[node_id]['prob']

    # if pca is True, reduce the dimensionality of the latent space to 2 dimensions
    # otherwise, use the first two dimensions of the latent space
    if pca:
        pca_fit = PCA(n_components=2)
        z_sample = pca_fit.fit_transform(z_sample)

    # plot the distribution plot at the given axes
    ax.hist(z_sample[:, 0], bins=50, color='darkblue', alpha=0.7)
    # ax.scatter(z_sample[:, 0], z_sample[:, 1], c=colors, cmap='tab10', alpha=weights, s = 0.25)
    ax.set_title(f"Node {node_id}")
    ax.set_xticks([])
    ax.set_yticks([])



def draw_tree_with_scatter_plots(data, node_embeddings, label_list, pca = True, dataset = None, flattened = False):
    """
    Draw the full tree graph with scatter plots as nodes. The scatter plots show the latent space of the node.

    Parameters
    ----------
    data: list
        The tree data
    node_embeddings: dict
        The node embeddings
    label_list: np.array
        The labels of the observations
    pca: bool
        Whether to use PCA to reduce the dimensionality of the latent space for visualization
    """

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in data:
        node_id, label, parent_id, node_type = node
        G.add_node(node_id, label=label, node_type=node_type)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    # Get positions of graph nodes
    pos = hierarchy_pos(G, 0, levels=None, width=1, height=1)

    # get the labels of the nodes, needed for the scatter plots legend
    labels = nx.get_node_attributes(G, 'label')

    # Create a figure
    fig, ax = plt.subplots(figsize=(20, 10))

    for node_id, node_data in G.nodes(data=True):
        x, y = pos[node_id]
        # Create a subplot for each node, centered on the node, and draw the scatter plot
        sub_ax = fig.add_axes([x, y+0.9, 0.1, 0.1])
        if flattened:
            draw_flattened_dist_node(node_id, node_embeddings, label_list, sub_ax, pca)
        else:
            draw_scatter_node(node_id, node_embeddings, label_list, sub_ax, pca)

    # Draw the lines between node plots
    for node in data:
        node_id, label, parent_id, node_type = node
        if parent_id is not None:
            # pick subplots of parent and child
            sub_ax_parent = fig.axes[parent_id + 1]
            sub_ax_child = fig.axes[node_id + 1]

            # draw the connection lines
            con = ConnectionPatch(xyA=(0.5, 0.5), xyB=(0.5, 0.5), coordsA='data', coordsB='data',
                                axesA=sub_ax_parent, axesB=sub_ax_child, color='black', alpha=0.5, zorder=-1)
            fig.add_artist(con)

    # Set the limits of the plot
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis('off')

    #### Legends

    # create a list of the unique labels
    if dataset == None:
        unique_labels = np.unique(label_list)
    elif dataset == 'mnist':
        unique_labels = np.arange(10)
    elif dataset == 'fmnist':
        unique_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    elif dataset == 'cifar10':
        unique_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    colors = plt.cm.tab10.colors
    patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                label="{:s}".format(str(unique_labels[i])) )[0]  for i in range(len(unique_labels)) ]
    
    # for each leaf node, plot a legend with the class labels if a class is present in node with frequency > 0.05
    for node in data:
        node_id, label, parent_id, node_type = node

        # pick subplot of node
        sub_ax = fig.axes[node_id + 1]

        # get the probabilities of the leaf node, only consider observactions with probability > 0.05 for a cluster
        prob = node_embeddings[node_id]['prob']
        labels = label_list[prob > 0.05]
        # count the number of class labels to get the frequency
        counts = np.unique(labels, return_counts=True)[1]
        counts = counts / np.sum(counts)
        labels = np.unique(labels, return_counts=True)[0]
        labels = labels[counts > 0.05]
        
        # create a list of the patches, select only labels that are in labels, but same color as unique_labels
        patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
            label=" ")[0]  for i in labels ]

        # plot the legend on right side of the subplot
        sub_ax.legend(handles=patches, bbox_to_anchor=(1.07, 0.5), frameon=False, fontsize=7, loc='center')

    # legend below the plot
    unique_labels = np.unique(label_list)
    colors = plt.cm.tab10.colors
    patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                label="{:s}".format(str(unique_labels[i])) )[0]  for i in range(len(unique_labels)) ]
    fig.legend(handles=patches, ncol=len(unique_labels), frameon=False, fontsize=15, loc='center', bbox_to_anchor=(0.5, 0.05))

    plt.show()


