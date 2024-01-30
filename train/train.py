import time
from pathlib import Path
import wandb
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import uuid
import os
import gc
import scipy
import yaml
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.data_utils import get_data, get_gen
from utils.utils import reset_random_seeds, cluster_acc, dendrogram_purity, leaf_purity, display_image
from utils.training_utils import compute_leaves, validate_one_epoch, Custom_Metrics, predict, move_to
from utils.model_utils import construct_data_tree
from train.train_tree import run_tree
from models.losses import loss_reconstruction_cov_mse_eval
from FID.fid_score import calculate_fid, get_precomputed_fid_scores_path, save_fid_stats, save_fid_stats_as_dict


def run_experiment(configs):
	# setting device on GPU if available, else CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#Additional Info when using cuda
	if device.type == 'cuda':
		print("Using", torch.cuda.get_device_name(0))
	elif device.type == 'mps':
		print("Using MPS")
	else:
		print("No GPU available")
	

	# Set paths
	project_dir = Path(__file__).absolute().parent
	timestr = time.strftime("%Y%m%d-%H%M%S")
	ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
	experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
	experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
	print("Experiment path: ", experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
	os.environ["WANDB_SILENT"] = "true"
	wandb.init(
		project="treevae-jorge",
		entity="mds-group",
		config=configs, 
		mode=configs['globals']['wandb_logging']
	)
	if configs['globals']['wandb_logging'] in ['online', 'disabled']:
		wandb.run.name = wandb.run.name.split("-")[-1] + "-"+ configs['run_name']
	elif configs['globals']['wandb_logging'] == 'offline':
		wandb.run.name = configs['run_name']
	else:
		raise ValueError('wandb needs to be set to online, offline or disabled.')

	# Reproducibility
	reset_random_seeds(configs['globals']['seed'])

	# Generate a new dataset each run
	trainset, trainset_eval, testset = get_data(configs)

	model = run_tree(trainset, trainset_eval, testset, configs, device)
	model.eval()

	# Save model
	if configs['globals']['save_model']:
		print("\nSaving weights at ", experiment_path)
		torch.save(model.state_dict(), experiment_path/'model_weights.pt')

	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)

	# Training set performance
	gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
	y_train = trainset_eval.dataset.targets[trainset_eval.indices].numpy()

	prob_leaves_train = predict(gen_train_eval, model, device, 'prob_leaves')
	_ = gc.collect()
	if configs['globals']['save_model']:
		with open(experiment_path / 'c_train.npy', 'wb') as save_file:
			np.save(save_file, prob_leaves_train)
	yy = np.squeeze(np.argmax(prob_leaves_train, axis=-1)).numpy()
	acc, idx = cluster_acc(y_train, yy, return_index=True)
	swap = dict(zip(range(len(idx)), idx))
	y_wandb = np.array([swap[i] for i in yy], dtype=np.uint8)
	wandb.log({"Train_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
																	 y_true=y_train, preds=y_wandb,
																	 class_names=range(len(idx)))})
	nmi = normalized_mutual_info_score(y_train, yy)
	ari = adjusted_rand_score(y_train, yy)
	wandb.log({"Train Accuracy": acc, "Train Normalized Mutual Information": nmi, "Train Adjusted Rand Index": ari})

	# Test set performance
	gen_test = get_gen(testset, configs, validation=True, shuffle=False)
	y_test = testset.dataset.targets[testset.indices].numpy()
		
	metrics_calc_test = Custom_Metrics(device)
	validate_one_epoch(gen_test, model, metrics_calc_test, 0, device, test=True)
	_ = gc.collect()

	node_leaves_test, prob_leaves_test = predict(gen_test, model, device, 'node_leaves','prob_leaves')
	_ = gc.collect()
	if configs['globals']['save_model']:
		with open(experiment_path / 'c_test.npy', 'wb') as save_file:
			np.save(save_file, prob_leaves_test)
	yy = np.squeeze(np.argmax(prob_leaves_test, axis=-1)).numpy()

	# Determine indeces of samples that fall into each leaf for DP&LP
	leaves = compute_leaves(model.tree)
	ind_samples_of_leaves = []
	for i in range(len(leaves)):
		ind_samples_of_leaves.append([leaves[i]['node'],np.where(yy==i)[0]])
	# Calculate leaf and dedrogram purity
	dp = dendrogram_purity(model.tree, y_test, ind_samples_of_leaves)
	lp = leaf_purity(model.tree, y_test, ind_samples_of_leaves)
	# Note: Only comparable DP & LP values wrt baselines if same n_leaves for all methods
	wandb.log({"Test Dendrogram Purity": dp, "Test Leaf Purity": lp})

	# Calculate confusion matrix, accuracy and nmi
	acc, idx = cluster_acc(y_test, yy, return_index=True)
	swap = dict(zip(range(len(idx)), idx))
	y_wandb = np.array([swap[i] for i in yy], dtype=np.uint8)
	wandb.log({"Test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
																	y_true=y_test, preds=y_wandb,
																	class_names=range(len(idx)))})
	nmi = normalized_mutual_info_score(y_test, yy)
	ari = adjusted_rand_score(y_test, yy)

	data_tree = construct_data_tree(model, y_predicted=yy, y_true=y_test, n_leaves=len(node_leaves_test),
									data_name=configs['data']['data_name'])

	if configs['globals']['save_model']:
		with open(experiment_path / 'data_tree.npy', 'wb') as save_file:
			np.save(save_file, data_tree)
		with open(experiment_path / 'config.yaml', 'w', encoding='utf8') as outfile:
			yaml.dump(configs, outfile, default_flow_style=False, allow_unicode=True)

	table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
	fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
	dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
	wandb.log({"dendogram_final": dendro})

	wandb.log({"Test Accuracy": acc, "Test Normalized Mutual Information": nmi, "Test Adjusted Rand Index": ari})
	print(np.unique(yy, return_counts=True))

	print("Accuracy:", acc)
	print("Normalized Mutual Information:", nmi)
	print("Adjusted Rand Index:", ari)
	print("Dendrogram Purity:", dp)
	print("Leaf Purity:", lp)
	print("Digits", np.unique(y_train))

	# save images
	if configs['training']['compute_fid']:
		print("\n" * 2)
		print("Saving images")

		n_imgs = 10
		num_leaves = len(node_leaves_test)

		# save generations
		with torch.no_grad():
			generations, p_c_z = model.generate_images(n_imgs, device)
		generations = move_to(generations, 'cpu')
		for i in range(n_imgs):
			if num_leaves == 1: # needed to avoid an error when plotting only one image
				fig, axs = plt.subplots(1, 1, figsize=(15, 2))
				axs.imshow(display_image(generations[0][i]), cmap=plt.get_cmap('gray'))
				axs.set_title(f"L0: " + f"p=%.2f" % torch.round(p_c_z[i][0], decimals=2))
				axs.axis('off')
			else:
				fig, axs = plt.subplots(1, num_leaves, figsize=(15, 2))
				for c in range(num_leaves):
					axs[c].imshow(display_image(generations[c][i]), cmap=plt.get_cmap('gray'))
					axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(p_c_z[i][c], decimals=2))
					axs[c].axis('off')
			# save image to wandb
			wandb.log({f"Generated Image": fig})
		_ = gc.collect()

		# save first n_imgs reconstructions from test set
		for i in range(n_imgs):
			inputs, labels = next(iter(gen_test))
			inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
			with torch.no_grad():
				reconstructions, node_leaves = model.compute_reconstruction(inputs_gpu)
			reconstructions = move_to(reconstructions, 'cpu')
			node_leaves = move_to(node_leaves, 'cpu')

			fig, axs = plt.subplots(1, num_leaves+1, figsize=(15, 2))
			axs[num_leaves].set_title(f"Class: {labels[i].item()}")
			axs[num_leaves].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
			axs[num_leaves].set_title("Original")
			axs[num_leaves].axis('off')
			for c in range(num_leaves):
				axs[c].imshow(display_image(reconstructions[c][i]), cmap=plt.get_cmap('gray'))
				axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(node_leaves[c]['prob'][i], decimals=2))
				axs[c].axis('off')
			# save image to wandb without label
			wandb.log({f"Reconstruction": fig})
		_ = gc.collect()

	# compute FID score
	if configs['training']['compute_fid']:
		print("\n" * 2)
		print("FID scores")

		# if FID/FID_stats_precomputed folder does not exist, create it
		if not os.path.exists("FID/fid_stats_precomputed"):
			os.makedirs("FID/fid_stats_precomputed")
		
		# Generations FID

		# generate 10k samples from the model
		n_imgs = 10000
		with torch.no_grad():
			generations, p_c_z = model.generate_images(n_imgs, device)

		# for each generated image, only save the ones that are in the leaf with the highest probability
		generations_list = []
		for i in range(n_imgs):  
			# only save generation from leaf with highest probability
			leaf_ind = torch.argmax(p_c_z[i])
			generations_list.append(generations[leaf_ind][i])
		gen_dataset = torch.stack(generations_list).squeeze()

		# compute FID score for generated images

		# precompute or load fid scores for train and test
		data_stats_train = get_precomputed_fid_scores_path(trainset.dataset.data, configs['data']['data_name'], subset="train", device=device)
		data_stats_test = get_precomputed_fid_scores_path(testset.dataset.data, configs['data']['data_name'], subset="test", device=device)
		# precompute fid scores for generated images
		stats_generations = save_fid_stats_as_dict(gen_dataset, batch_size=256, device=device, dims=2048)
		train_FID_generations = calculate_fid([data_stats_train, stats_generations], batch_size=256, device=device, dims=2048)
		test_FID_generations = calculate_fid([data_stats_test, stats_generations], batch_size=256, device=device, dims=2048)
		print("FID score for generated images compared to train set:", train_FID_generations)
		print("FID score for generated images compared to test set:", test_FID_generations)

		wandb.log({"train_FID_generations": train_FID_generations, "test_FID_generations": test_FID_generations})
		_ = gc.collect()


		# Reconstructions FID

		for subset in ['train', 'test']:
			reconstructions_list = []

			if subset == 'train':
				data_loader = gen_train_eval
			elif subset == 'test':
				data_loader = gen_test

			for inputs, labels in tqdm(data_loader):
				inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
				with torch.no_grad():
					reconstructions, node_leaves = model.compute_reconstruction(inputs_gpu)
				_ = gc.collect()
				    
				# add reconstruction to list
				for i in range(len(inputs)):
					# probs are the probabilities of each leaf for the data point i
					probs = [node_leaves[j]['prob'][i] for j in range(len(node_leaves))]
					# use the leaf with highest probability
					leaf_ind = torch.argmax(torch.tensor(probs))
					# add reconstruction to list
					reconstructions_list.append(reconstructions[leaf_ind][i])
			reconstructions_dataset = torch.stack(reconstructions_list).squeeze()
			
			# precompute fid scores for generated images
			stats_reconstructions = save_fid_stats_as_dict(reconstructions_dataset, batch_size=256, device=device, dims=2048)

			if subset == 'train':
				train_FID_reconstructions = calculate_fid([data_stats_train, stats_reconstructions], batch_size=256, device=device, dims=2048)
				print("FID score for reconstructed images, train set:", train_FID_reconstructions)
			elif subset == 'test':
				test_FID_reconstructions = calculate_fid([data_stats_test, stats_reconstructions], batch_size=256, device=device, dims=2048)
				print("FID score for reconstructed images, test set:", test_FID_reconstructions)
			_ = gc.collect()

		wandb.log({"train_FID_reconstructions": train_FID_reconstructions, "test_FID_reconstructions": test_FID_reconstructions})


	# Compute the log-likehood
	if configs['training']['compute_ll']:
		if configs['training']['activation'] == 'sigmoid':
			print('\nComputing the log likelihood.... it might take a while.')
			ESTIMATION_SAMPLES = 1000
			elbo = np.zeros((len(testset), ESTIMATION_SAMPLES)) 
			for j in tqdm(range(ESTIMATION_SAMPLES)):
				elbo[:, j] = predict(gen_test, model, device, 'elbo')
				_ = gc.collect()
			elbo_new = elbo[:, :ESTIMATION_SAMPLES]
			log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
			marginal_log_likelihood = np.sum(log_likel) / len(testset)
			wandb.log({"test log-likelihood": marginal_log_likelihood})
			print("Test log-likelihood", marginal_log_likelihood)
			output_elbo, output_rec_loss = predict(gen_test, model, device, 'elbo', 'rec_loss')
			# ELBO
			print('Test ELBO:', -torch.mean(output_elbo))
			# Reconst. Error:
			print('Test Reconstruction Loss:', torch.mean(output_rec_loss))

		elif configs['training']['activation'] == 'mse':
			# Correct calculation of ELBO and Loglikelihood for 3channel images without assuming diagonal gaussian for reconstruction
			old_loss = model.loss
			model.loss = loss_reconstruction_cov_mse_eval
			# Note that for comparability to other papers, one might want to add Uniform(0,1) noise to the input images (in 0,255), 
			# to go from the discrete to the assumed continuous inputs 
		#    x_test_elbo = x_test * 255
		#    x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
			output_elbo, output_rec_loss = predict(gen_test, model, device, 'elbo', 'rec_loss')
			nelbo = torch.mean(output_elbo)
			nelbo_bpd = nelbo/(torch.log(2.)*testset.dataset.data.size()[1:].numel()) + 8 # Add 8 to account normalizing of inputs
			model.loss = old_loss
			elbo = np.zeros((len(testset), ESTIMATION_SAMPLES))
			for j in range(ESTIMATION_SAMPLES):
				#x_test_elbo = x_test * 255
				#x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
				output_elbo = predict(gen_test, model, device, 'elbo')
				elbo[:, j] = output_elbo
			# Change to bpd
			elbo_new = elbo[:, :ESTIMATION_SAMPLES]
			log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
			marginal_log_likelihood = np.sum(log_likel) / len(testset)
			marginal_log_likelihood = marginal_log_likelihood/(torch.log(2.)*testset.dataset.data.size()[1:].numel()) - 8
			print('Test Log-Likelihood Bound:', marginal_log_likelihood)
			# ELBO
			print('Test ELBO:', -nelbo_bpd)
			# Reconst. Error:
			print('Test Reconstruction Loss:', torch.mean(output_rec_loss)/(torch.log(2.)*testset.dataset.data.size()[1:].numel())+ 8)
			model.loss = old_loss
		else: 
			raise NotImplementedError
		
			
	wandb.finish(quiet=True)
