import numpy
import math
from itertools import islice
import attr
import math
from scipy.special import comb
import os
from utils.utils import merge_yaml_args
from utils.training_utils import move_to
from baselines.lvae.losses import loss_reconstruction_binary, loss_reconstruction_mse
from pathlib import Path
import yaml
import torch
import wandb
from tqdm import tqdm
from torchmetrics import Metric



def window(seq, n):
	"Returns a sliding window (of width n) over data from the iterable"
	"   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
	it = iter(seq)
	result = tuple(islice(it, n))
	if len(result) == n:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result

def count_values_in_sequence(seq):
	from collections import defaultdict
	res = defaultdict(lambda : 0)
	for key in seq:
		res[key] += 1
	return dict(res)

def weighted_avg_and_std(values, weights):
	"""
	Return the weighted average and standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""
	#https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
	average = numpy.average(values, weights=weights)
	# Fast and numerically precise:
	variance = numpy.average((values-average)**2, weights=weights)
	return (average, math.sqrt(variance))
# The dendogram purity can be computed in a bottom up manner
# at each node within the tree we need the number of data points in each child and have to weight the cases


@attr.s(cmp=False)
class DpNode(object):
	"""
	node_id should be in such a way that a smaller number means split before a larger number in a top-down manner
	That is the root should have node_id = 0 and the children of the last split should have node id
	2*n_dps-2 and 2*n_dps-1

	"""

	left_child = attr.ib()
	right_child = attr.ib()
	node_id = attr.ib()

	@property
	def children(self):
		return [self.left_child, self.right_child]

	@property
	def is_leaf(self):
		return False


@attr.s(cmp=False)
class DpLeaf(object):
	dp_ids = attr.ib()
	node_id = attr.ib()

	@property
	def children(self):
		return []

	@property
	def is_leaf(self):
		return True


def combine_to_trees(tree_a, tree_b):
	def recursive(ta, tb):
		if ta.is_leaf != tb.is_leaf or ta.node_id != tb.node_id:
			print(f"{ta.node_id} != {tb.node_id}")
			raise RuntimeError("Trees are not equivalent!")
		if ta.is_leaf:
			return DpLeaf(ta.dp_ids + tb.dp_ids, ta.node_id)
		else:
			left_child = recursive(ta.left_child, tb.left_child)
			right_child = recursive(ta.right_child, tb.right_child)
			return DpNode(left_child, right_child, ta.node_id)

	return recursive(tree_a, tree_b)


def leaf_purity(tree_root, ground_truth):
	values = []
	weights = []

	def get_leaf_purities(node):
		nonlocal values
		nonlocal weights
		if node.is_leaf:
			node_total_dp_count = len(node.dp_ids)
			node_per_label_counts = count_values_in_sequence(
				[ground_truth[id] for id in node.dp_ids])
			if node_total_dp_count > 0:
				purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
			else:
				purity_rate = 1.0
			values.append(purity_rate)
			weights.append(node_total_dp_count)
		else:
			get_leaf_purities(node.left_child)
			get_leaf_purities(node.right_child)

	get_leaf_purities(tree_root)

	return weighted_avg_and_std(values, weights)


def dendrogram_purity(tree_root, ground_truth):
	total_per_label_frequencies = count_values_in_sequence(ground_truth)
	total_per_label_pairs_count = {k: comb(v, 2, True) for k, v in total_per_label_frequencies.items()}
	total_n_of_pairs = sum(total_per_label_pairs_count.values())

	one_div_total_n_of_pairs = 1. / total_n_of_pairs

	purity = 0.

	def calculate_purity(node, level):
		nonlocal purity
		if node.is_leaf:
			node_total_dp_count = len(node.dp_ids)
			node_per_label_frequencies = count_values_in_sequence(
				[ground_truth[id] for id in node.dp_ids])
			node_per_label_pairs_count = {k: comb(v, 2, True) for k, v in node_per_label_frequencies.items()}

		else:  # it is an inner node
			left_child_per_label_freq, left_child_total_dp_count = calculate_purity(node.left_child, level + 1)
			right_child_per_label_freq, right_child_total_dp_count = calculate_purity(node.right_child, level + 1)
			node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
			node_per_label_frequencies = {k: left_child_per_label_freq.get(k, 0) + right_child_per_label_freq.get(k, 0) \
										  for k in set(left_child_per_label_freq) | set(right_child_per_label_freq)}

			node_per_label_pairs_count = {k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k) \
										  for k in set(left_child_per_label_freq) & set(right_child_per_label_freq)}

		for label, pair_count in node_per_label_pairs_count.items():
			label_freq = node_per_label_frequencies[label]
			label_pairs = node_per_label_pairs_count[label]
			purity += one_div_total_n_of_pairs * label_freq / node_total_dp_count * label_pairs
		return node_per_label_frequencies, node_total_dp_count

	calculate_purity(tree_root, 0)
	return purity


def prune_dendrogram_purity_tree(tree, n_leaves):
	"""
	This function collapses the tree such that it only has n_leaves.
	This makes it possible to compare different trees with different number of leaves.

	Important, it assumes that the node_id is equal to the split order, that means the tree root should have the smallest split number
	and the two leaf nodes that are splitted the last have the highest node id. And that  max(node_id) == #leaves - 2

	:param tree:
	:param n_levels:
	:return:
	"""
	max_node_id = n_leaves - 2

	def recursive(node):
		if node.is_leaf:
			return node
		else:  # node is an inner node
			if node.node_id < max_node_id:
				left_child = recursive(node.left_child)
				right_child = recursive(node.right_child)
				return DpNode(left_child, right_child, node.node_id)
			else:
				work_list = [node.left_child, node.right_child]
				dp_ids = []
				while len(work_list) > 0:
					nc = work_list.pop()
					if nc.is_leaf:
						dp_ids = dp_ids + nc.dp_ids
					else:
						work_list.append(nc.left_child)
						work_list.append(nc.right_child)
				return DpLeaf(dp_ids, node.node_id)
				# raise RuntimeError("should not be here!")

	return recursive(tree)


def to_dendrogram_purity_tree(children_array):
	"""
	Can convert the children_ matrix of a  sklearn.cluster.hierarchical.AgglomerativeClustering outcome to a dendrogram_purity tree
	:param children_array:  array-like, shape (n_samples-1, 2)
		The children of each non-leaf nodes. Values less than `n_samples`
			correspond to leaves of the tree which are the original samples.
			A node `i` greater than or equal to `n_samples` is a non-leaf
			node and has children `children_[i - n_samples]`. Alternatively
			at the i-th iteration, children[i][0] and children[i][1]
			are merged to form node `n_samples + i`
	:return:
	"""
	n_samples = children_array.shape[0] + 1
	max_id = 2 * n_samples - 2
	node_map = {dp_id: DpLeaf([dp_id], max_id - dp_id) for dp_id in range(n_samples)}
	next_id = max_id - n_samples

	for idx in range(n_samples - 1):
		next_fusion = children_array[idx, :]
		child_a = node_map.pop(next_fusion[0])
		child_b = node_map.pop(next_fusion[1])
		node_map[n_samples + idx] = DpNode(child_a, child_b, next_id)
		next_id -= 1
	if len(node_map) != 1:
		raise RuntimeError("tree must be fully developed! Use ompute_full_tree=True for AgglomerativeClustering")
	root = node_map[n_samples + n_samples - 2]
	return root

def prepare_config(args, project_dir):
	data_name = args.config_name +'.yml'
	config_path = project_dir / 'configs' / data_name
	print(config_path)

	with config_path.open(mode='r') as yamlfile:
		configs = yaml.safe_load(yamlfile)

	# Override config if args in parser
	configs = merge_yaml_args(configs, args)

	if isinstance(configs['training']['latent_dim'], str):
		a = configs['training']['latent_dim'].split(",")
		configs['training']['latent_dim'] = [int(i) for i in a]
	if isinstance(configs['training']['mlp_layers'], str):
		a = configs['training']['mlp_layers'].split(",")
		configs['training']['mlp_layers'] = [int(i) for i in a]
	a = configs['training']['augmentation_method'].split(",")
	configs['training']['augmentation_method'] = [str(i) for i in a]
	configs['globals']['results_dir'] = os.path.join(project_dir, 'models/experiments')
	configs['globals']['results_dir'] = Path(configs['globals']['results_dir']).absolute()
	if configs['training']['activation'] == "sigmoid":
		loss = loss_reconstruction_binary 
	elif configs['training']['activation'] == "mse":
		loss = loss_reconstruction_mse
				
	return configs, loss


def train_one_epoch(train_loader, model, optimizer, metrics_calc, epoch_idx, device):
	model.train()
	metrics_calc.reset()

	for batch_idx, batch in enumerate(tqdm(train_loader, leave=False)):
		inputs, labels = batch
		inputs, labels = inputs.to(device), labels.to(device)
		# Zero your gradients for every batch
		optimizer.zero_grad()

		# Make predictions for this batch
		outputs = model(inputs)

		# Compute the loss and its gradients
		rec_loss = outputs['rec_loss']
		kl_losses = outputs['kl_root'] + outputs['kl_nodes']
		
		loss_value = rec_loss + model.alpha * kl_losses + outputs['aug_decisions']
		loss_value.backward()

		# Adjust learning weights
		optimizer.step()

		# Store metrics
		metrics_calc.update(outputs['elbo_samples'].numel(), loss_value, outputs['rec_loss'], outputs['kl_nodes'], outputs['kl_root'],outputs['aug_decisions'])
	
	# Calculate and log metrics
	metrics = metrics_calc.compute()
	metrics['alpha'] = model.alpha
	wandb.log({'train': metrics})
	prints = f"Epoch {epoch_idx}, Train     : "
	for key, value in metrics.items():
		prints += f"{key}: {value:.3f} "
	print(prints)
	return


def validate_one_epoch(test_loader, model, metrics_calc, epoch_idx, device, small_tree=False, test=False):
	model.eval()
	metrics_calc.reset()

	with torch.no_grad():
		for batch_idx, batch in enumerate(tqdm(test_loader, leave=False)):
			inputs, labels = batch
			inputs, labels = inputs.to(device), labels.to(device)

			# Make predictions for this batch
			outputs = model(inputs)

			# Compute the loss and its gradients
			rec_loss = outputs['rec_loss']
			kl_losses = outputs['kl_root'] + outputs['kl_nodes']

			loss_value = rec_loss + model.alpha * kl_losses + outputs['aug_decisions']


			# Store metrics
			metrics_calc.update(outputs['elbo_samples'].numel(), loss_value, outputs['rec_loss'], outputs['kl_nodes'], outputs['kl_root'],outputs['aug_decisions'])
	# Calculate and log metrics
	metrics = metrics_calc.compute()
	if not test:
		wandb.log({'validation': metrics})
		prints = f"Epoch {epoch_idx}, Validation: "
	else: 
		wandb.log({'test': metrics})
		prints = f"Test: "
	for key, value in metrics.items():
		prints += f"{key}: {value:.3f} "
	print(prints)
	return


class Custom_Metrics(Metric):
	def __init__(self, device):
		super().__init__()
		self.add_state("loss_value", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
		self.add_state("rec_loss", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
		self.add_state("kl_root", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
		self.add_state("kl_nodes", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
		self.add_state("aug_decisions", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
		self.add_state("n_samples", default=torch.tensor(0, dtype=torch.int, device=device), dist_reduce_fx="sum")


	def update(self, n_samples: torch.Tensor, loss_value: torch.Tensor, rec_loss: torch.Tensor, kl_nodes: torch.Tensor,  kl_root: torch.Tensor, 
		   aug_decisions: torch.Tensor):

		n_samples = n_samples
		self.n_samples += n_samples
		self.loss_value += loss_value.item()*n_samples
		self.rec_loss += rec_loss.item()*n_samples
		self.kl_root += kl_root.item()*n_samples
		self.kl_nodes += kl_nodes.item()*n_samples
		self.aug_decisions += aug_decisions.item()*n_samples


	def compute(self):
		metrics = dict({'loss_value': self.loss_value/self.n_samples, 'rec_loss': self.rec_loss/self.n_samples, 'kl_root': self.kl_root/self.n_samples,
		  				'kl_nodes': self.kl_nodes/self.n_samples, 'aug_decisions': self.aug_decisions/self.n_samples})

		return metrics
	

def predict_elbo(loader, model, device):
	model.eval()

	elbo_samples = []

	with torch.no_grad():
		for batch_idx, (inputs, labels) in enumerate(loader):
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			elbo_samples.append(move_to(outputs['elbo_samples'], 'cpu'))

	elbo_samples = torch.cat(elbo_samples,dim=0)
	return elbo_samples