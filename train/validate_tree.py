"""
Evaluate the performance of a trained TreeVAE model on both the train and test datasets.
"""
import wandb
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import gc
import yaml
import torch
import scipy
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.data_utils import get_gen
from utils.utils import cluster_acc, dendrogram_purity, leaf_purity, display_image
from utils.training_utils import compute_leaves, validate_one_epoch, Custom_Metrics, predict, move_to
from utils.model_utils import construct_data_tree
from models.losses import loss_reconstruction_cov_mse_eval
from FID.fid_score import calculate_fid, get_precomputed_fid_scores_path, save_fid_stats_as_dict


def val_tree(trainset, testset, model, device, experiment_path, configs):
    """
    Run the validation of a trained instance of TreeVAE on both the train and test datasets. All final results and
    validations will be stored in Wandb, while the most important ones will be also printed out in the terminal.

    Parameters
    ----------
    trainset: torch.utils.data.Dataset
        The train dataset
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    experiment_path: str
        The experimental path where to store the tree
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """

    ############ Training set performance ############

    # get the data loader
    gen_train_eval = get_gen(trainset, configs, validation=True, shuffle=False)
    y_train = trainset.dataset.targets[trainset.indices].numpy()
    # compute the leaf probabilities
    prob_leaves_train = predict(gen_train_eval, model, device, 'prob_leaves')
    _ = gc.collect()
    # compute the predicted cluster
    y_train_pred = np.squeeze(np.argmax(prob_leaves_train, axis=-1)).numpy()
    # compute clustering metrics
    acc, idx = cluster_acc(y_train, y_train_pred, return_index=True)
    nmi = normalized_mutual_info_score(y_train, y_train_pred)
    ari = adjusted_rand_score(y_train, y_train_pred)
    wandb.log({"Train Accuracy": acc, "Train Normalized Mutual Information": nmi, "Train Adjusted Rand Index": ari})
    # compute confusion matrix
    swap = dict(zip(range(len(idx)), idx))
    y_wandb = np.array([swap[i] for i in y_train_pred], dtype=np.uint8)
    wandb.log({"Train_confusion_matrix":
                   wandb.plot.confusion_matrix(probs=None, y_true=y_train, preds=y_wandb, class_names=range(len(idx)))})

    ############ Test set performance ############

    # get the data loader
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    y_test = testset.dataset.targets[testset.indices].numpy()
    # compute one validation pass through the test set to log losses
    metrics_calc_test = Custom_Metrics(device)
    validate_one_epoch(gen_test, model, metrics_calc_test, 0, device, test=True)
    _ = gc.collect()
    # predict the leaf probabilities and the leaves
    node_leaves_test, prob_leaves_test = predict(gen_test, model, device, 'node_leaves', 'prob_leaves')
    _ = gc.collect()
    # compute the predicted cluster
    y_test_pred = np.squeeze(np.argmax(prob_leaves_test, axis=-1)).numpy()
    # Calculate clustering metrics
    acc, idx = cluster_acc(y_test, y_test_pred, return_index=True)
    nmi = normalized_mutual_info_score(y_test, y_test_pred)
    ari = adjusted_rand_score(y_test, y_test_pred)
    wandb.log({"Test Accuracy": acc, "Test Normalized Mutual Information": nmi, "Test Adjusted Rand Index": ari})
    # Calculate confusion matrix
    swap = dict(zip(range(len(idx)), idx))
    y_wandb = np.array([swap[i] for i in y_test_pred], dtype=np.uint8)
    wandb.log({"Test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                    y_true=y_test, preds=y_wandb,
                                                                    class_names=range(len(idx)))})

    # Determine indices of samples that fall into each leaf for Dendogram Purity & Leaf Purity
    leaves = compute_leaves(model.tree)
    ind_samples_of_leaves = []
    for i in range(len(leaves)):
        ind_samples_of_leaves.append([leaves[i]['node'], np.where(y_test_pred == i)[0]])
    # Calculate leaf and dedrogram purity
    dp = dendrogram_purity(model.tree, y_test, ind_samples_of_leaves)
    lp = leaf_purity(model.tree, y_test, ind_samples_of_leaves)
    # Note: Only comparable DP & LP values wrt baselines if they have the same n_leaves for all methods
    wandb.log({"Test Dendrogram Purity": dp, "Test Leaf Purity": lp})

    # Save the tree structure of TreeVAE and log it
    data_tree = construct_data_tree(model, y_predicted=y_test_pred, y_true=y_test, n_leaves=len(node_leaves_test),
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

    # Printing important results
    print(np.unique(y_test_pred, return_counts=True))
    print("Accuracy:", acc)
    print("Normalized Mutual Information:", nmi)
    print("Adjusted Rand Index:", ari)
    print("Dendrogram Purity:", dp)
    print("Leaf Purity:", lp)
    print("Digits", np.unique(y_test))

    # Compute the log-likehood of the test data
    # ATTENTION it might take a while! If not interested disable the setting in configs
    if configs['training']['compute_ll']:
        compute_likelihood(testset, model, device, configs)

    # Save images from the test set to wandb
    if configs['training']['save_images']:
        save_images(10, len(node_leaves_test), testset, model, device, configs)

    # Compute FID scores
    if configs['training']['compute_fid']:
        compute_FID_scores(trainset, testset, model, device, configs)

    return


def compute_likelihood(testset, model, device, configs):
    """
    Compute the approximated log-likelihood calculated using 1000 importance-weighted samples.

    Parameters
    ----------
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """
    ESTIMATION_SAMPLES = 1000
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    print('\nComputing the log likelihood.... it might take a while.')
    if configs['training']['activation'] == 'sigmoid':
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
        print('Test ELBO:', -torch.mean(output_elbo))
        print('Test Reconstruction Loss:', torch.mean(output_rec_loss))

    elif configs['training']['activation'] == 'mse':
        # Correct calculation of ELBO and Loglikelihood for 3channel images without assuming diagonal gaussian for
        # reconstruction
        old_loss = model.loss
        model.loss = loss_reconstruction_cov_mse_eval
        # Note that for comparability to other papers, one might want to add Uniform(0,1) noise to the input images
        # (in 0,255), to go from the discrete to the assumed continuous inputs
        #    x_test_elbo = x_test * 255
        #    x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
        output_elbo, output_rec_loss = predict(gen_test, model, device, 'elbo', 'rec_loss')
        nelbo = torch.mean(output_elbo)
        nelbo_bpd = nelbo / (torch.log(torch.tensor(2)) * configs['training']['inp_shape']) + 8  # Add 8 to account normalizing of inputs
        model.loss = old_loss
        elbo = np.zeros((len(testset), ESTIMATION_SAMPLES))
        for j in range(ESTIMATION_SAMPLES):
            # x_test_elbo = x_test * 255
            # x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
            output_elbo = predict(gen_test, model, device, 'elbo')
            elbo[:, j] = output_elbo
        # Change to bpd
        elbo_new = elbo[:, :ESTIMATION_SAMPLES]
        log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
        marginal_log_likelihood = np.sum(log_likel) / len(testset)
        marginal_log_likelihood = marginal_log_likelihood / (
                torch.log(torch.tensor(2)) * configs['training']['inp_shape']) - 8
        wandb.log({"test log-likelihood": marginal_log_likelihood})
        print('Test Log-Likelihood Bound:', marginal_log_likelihood)
        print('Test ELBO:', -nelbo_bpd)
        print('Test Reconstruction Loss:',
              torch.mean(output_rec_loss) / (torch.log(torch.tensor(2)) * configs['training']['inp_shape']) + 8)
        model.loss = old_loss
    else:
        raise NotImplementedError
    return


def compute_FID_scores(trainset, testset, model, device, configs):
    """
    Compute the FID scores for the train and test sets and for the generated and reconstructed images.

    Parameters
    ----------
    trainset: torch.utils.data.Dataset
        The train dataset
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """

    print("\n" * 2)
    print("FID scores")

    # if FID/FID_stats_precomputed folder does not exist, create it
    if not os.path.exists("FID/fid_stats_precomputed"):
        os.makedirs("FID/fid_stats_precomputed")

    # precompute or load fid scores for train and test
    if configs['data']['data_name'] in ["mnist", "fmnist", "cifar10"]:
        data_stats_train = get_precomputed_fid_scores_path(trainset.dataset.data, configs['data']['data_name'],
                                                       batch_size=50, subset="train", device=device)
        data_stats_test = get_precomputed_fid_scores_path(testset.dataset.data, configs['data']['data_name'],
                                                      batch_size=50, subset="test", device=device)
    elif configs['data']['data_name'] in ["celeba", "cubicc"]:

        train_data_set = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
                                     torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=False)])
        test_data_set = np.concatenate([batch[0].permute(0, 2, 3, 1).numpy() for batch in
                                    torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)])
        data_stats_train = get_precomputed_fid_scores_path(train_data_set, configs['data']['data_name'],
                                                       batch_size=50, subset="train", device=device)
        data_stats_test = get_precomputed_fid_scores_path(test_data_set, configs['data']['data_name'],
                                                      batch_size=50, subset="test", device=device)

    # Generations FID -----------------------------------------------------------

    # generate 10k samples from the model
    n_imgs = 10000
    with torch.no_grad():
        generations, p_c_z = model.generate_images(n_imgs, device)
    generations = move_to(generations, 'cpu')
    p_c_z = move_to(p_c_z, 'cpu')

    # for each generated image, only save the ones that are in the leaf with the highest probability
    generations_list = []
    for i in range(n_imgs):
        # only save generation from leaf with highest probability
        leaf_ind = torch.argmax(p_c_z[i])
        generations_list.append(generations[leaf_ind][i])
    gen_dataset = torch.stack(generations_list).squeeze()
    _ = gc.collect()

    # compute FID score for generated images

    # precompute FID scores for generated images
    stats_generations = save_fid_stats_as_dict(gen_dataset, batch_size=50, device=device, dims=2048)
    train_FID_generations = calculate_fid([data_stats_train, stats_generations], batch_size=50, device=device,
                                          dims=2048)
    test_FID_generations = calculate_fid([data_stats_test, stats_generations], batch_size=50, device=device, dims=2048)
    print("FID score for generated images compared to train set:", train_FID_generations)
    print("FID score for generated images compared to test set:", test_FID_generations)

    wandb.log({"train_FID_generations": train_FID_generations, "test_FID_generations": test_FID_generations})
    _ = gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reconstructions FID -----------------------------------------------------------

    # only compute reconstruction FID for test set in colored images to reduce cuda memory usage
    if configs['data']['data_name'] == "cifar10":
        fid_eval_set = ['test']
    else:
        fid_eval_set = ['train', 'test']

    for subset in fid_eval_set:
        reconstructions_list = []

        if subset == 'train':
            gen_train_eval = get_gen(trainset, configs, validation=True, shuffle=False)
            data_loader = gen_train_eval
        elif subset == 'test':
            gen_test = get_gen(testset, configs, validation=True, shuffle=False)
            data_loader = gen_test

        for inputs, labels in tqdm(data_loader):
            inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
            with torch.no_grad():
                reconstructions, node_leaves, _ = model.compute_reconstruction(inputs_gpu)
            reconstructions = move_to(reconstructions, 'cpu')
            node_leaves = move_to(node_leaves, 'cpu')
            _ = gc.collect()

            # add reconstruction to list
            for i in range(len(inputs)):
                # probs are the probabilities of each leaf for the data point i
                probs = [node_leaves[j]['prob'][i] for j in range(len(node_leaves))]
                # use the leaf with highest probability
                leaf_ind = torch.argmax(torch.tensor(probs))
                # add reconstruction to list
                reconstructions_list.append(reconstructions[leaf_ind][i])
        reconstructions_dataset = torch.stack(reconstructions_list).squeeze().detach()
        _ = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if configs['data']['data_name'] == "cifar10":
            batch_size_fid = 25
        else:
            batch_size_fid = 50

        # precompute FID scores for generated images
        stats_reconstructions = save_fid_stats_as_dict(reconstructions_dataset, batch_size=batch_size_fid, device=device,
                                                       dims=2048)
        _ = gc.collect()

        if subset == 'train':
            train_FID_reconstructions = calculate_fid([data_stats_train, stats_reconstructions], batch_size=batch_size_fid,
                                                      device=device, dims=2048)
            print("FID score for reconstructed images, train set:", train_FID_reconstructions)
            wandb.log({"train_FID_reconstructions": train_FID_reconstructions})
        elif subset == 'test':
            test_FID_reconstructions = calculate_fid([data_stats_test, stats_reconstructions], batch_size=batch_size_fid,
                                                     device=device, dims=2048)
            print("FID score for reconstructed images, test set:", test_FID_reconstructions)
            wandb.log({"test_FID_reconstructions": test_FID_reconstructions})
        _ = gc.collect()

    return


def save_images(n_imgs, num_leaves, testset, model, device, configs):
    """
    Save images from the test set to wandb.

    Parameters
    ----------
    n_imgs: int
        The number of images to save
    num_leaves: int
        The number of leaves in the tree
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """
    print("\n" * 2)
    print("Saving images")

    gen_test = get_gen(testset, configs, validation=True, shuffle=False)

    # save n_imgs generations
    with torch.no_grad():
        generations, p_c_z = model.generate_images(n_imgs, device)
    generations = move_to(generations, 'cpu')
    for i in range(n_imgs):
        if num_leaves == 1:  # needed to avoid an error when plotting only one image
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
            reconstructions, node_leaves, _ = model.compute_reconstruction(inputs_gpu)
        reconstructions = move_to(reconstructions, 'cpu')
        node_leaves = move_to(node_leaves, 'cpu')

        fig, axs = plt.subplots(1, num_leaves + 1, figsize=(15, 2))
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
    return
