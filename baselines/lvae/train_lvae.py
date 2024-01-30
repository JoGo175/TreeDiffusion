import time
from pathlib import Path
import uuid
import os
import gc
import wandb
import numpy as np
import scipy
import torch
import torch.optim as optim
from utils.training_utils import AnnealKLCallback, get_optimizer
from utils.data_utils import get_data, get_gen
from utils.utils import reset_random_seeds
from baselines.lvae.model_lvae import LadderVAE
from baselines.utils_tree import Custom_Metrics, train_one_epoch, validate_one_epoch, predict_elbo


def run_experiment(configs):
	# setting device on GPU if available, else CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#Additional Info when using cuda
	if device.type == 'cuda':
		print("Using", torch.cuda.get_device_name(0))
	else:
		print("No GPU available")

	# Set paths
	project_dir = Path(__file__).absolute().parent
	timestr = time.strftime("%Y%m%d-%H%M%S")
	ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
	experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
	experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../../models/logs', ex_name))
	print(experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../../wandb', '.cache', 'wandb')
	os.environ["WANDB_SILENT"] = "true"
	wandb.init(
		project="Pytorch-debugging",
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

	gen_train = get_gen(trainset, configs, validation=False, shuffle=True)
	gen_test = get_gen(testset, configs, validation=True, shuffle=False)

	# Define model & optimizer
	_ = gc.collect()
	model = LadderVAE(**configs['training'])
	model.to(device)

	if not configs['globals']['eager_mode']:
		model = torch.compile(model)

	optimizer = get_optimizer(model, configs)

	# Initialize schedulers
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'], gamma=configs['training']['decay_lr'])
	alpha_scheduler = AnnealKLCallback(model, decay=configs['training']['decay_kl'], start=configs['training']['kl_start'])

	# Initialize Metrics
	metrics_calc_train = Custom_Metrics(device)
	metrics_calc_val = Custom_Metrics(device)

	for epoch in range(configs['training']['num_epochs']):  # loop over the dataset multiple times
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device)
		lr_scheduler.step()
		alpha_scheduler.on_epoch_end(epoch)
	_ = gc.collect()


	print('\n*****************model finetuning******************\n')
	# Initialize optimizer and schedulers
	optimizer = get_optimizer(model, configs)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'], gamma=configs['training']['decay_lr'])
	model.alpha=1
	
	for epoch in range(configs['training']['num_epochs_finetuning']):  # loop over the dataset multiple times
		assert model.alpha == 1.
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device)
		lr_scheduler.step()
	_ = gc.collect()


	model.eval()

	# Save model
	if configs['globals']['save_model']:
		print("\nSaving weights at ", experiment_path)
		torch.save(model.state_dict(), experiment_path/'model_weights.pt')

	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)

	_ = gc.collect()

	# Test set performance
	metrics_calc_test = Custom_Metrics(device)
	validate_one_epoch(gen_test, model, metrics_calc_test, 0, device, test=True) 
	_ = gc.collect()

	# Compute the log-likehood
	if configs['training']['compute_ll']:
		print('\nComputing the log likelihood.... it might take a while.')
		ESTIMATION_SAMPLES = 1000
		elbo = np.zeros((len(testset), ESTIMATION_SAMPLES))
		for j in range(ESTIMATION_SAMPLES):
			elbo[:, j] = predict_elbo(gen_test, model, device)
			_ = gc.collect()
		elbo_new = elbo[:, :ESTIMATION_SAMPLES]
		log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
		marginal_log_likelihood = np.sum(log_likel) / len(testset)
		wandb.log({"test log-likelihood": marginal_log_likelihood})
		print("Test log-likelihood", marginal_log_likelihood)

	wandb.finish(quiet=True)
