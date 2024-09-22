# ---------------------------------------------------------------
#
# This file has been modified from a file in the pytorch_fid library
# which was released under the Apache License v2.0 License.
#
# Source:
# https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_pytorch_fid). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------


"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from FID.inception import InceptionV3

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def convert_images_to_rgb(images_tensor):
    """
    Converts a tensor of images to RGB format (if not already in RGB format).
    """
    # set images to tensor if not already
    if not isinstance(images_tensor, torch.Tensor):
        images_tensor = torch.tensor(images_tensor)

    # for RGB images, check if channel is first or last, set it to first
    if len(images_tensor.shape) == 4:
        if images_tensor.shape[3] == 3:
            images_tensor = images_tensor.permute(0, 3, 1, 2)

    # grayscale images have no color channel --> add it
    if len(images_tensor.shape) == 3:
        images_tensor = images_tensor.unsqueeze(1)

    num_images, num_channels, _, _ = images_tensor.shape

    # convert to RGB
    rgb_images = []
    for i in range(num_images):
        img = TF.functional.to_pil_image(images_tensor[i])
        img = img.convert('RGB')
        rgb_images.append(img)
    rgb_images = np.stack(rgb_images)

    return rgb_images


class ImagePathDataset(torch.utils.data.Dataset):
    """
    Dataset class given dataset of images.
    """

    def __init__(self, images, transforms=None):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(dataset, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataset     : Image dataset
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # prepare dataloader
    dataset = convert_images_to_rgb(dataset)
    dataset = ImagePathDataset(dataset, transforms=TF.ToTensor())

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)

    # compute and return activations
    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2)
    is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(data, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- data       : Dataset of images
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path_or_data, model, batch_size, dims, device, num_workers=1):
    """
    Loads or computes the FID statistics for a dataset of images

    Params:
    -- path_or_data : image dataset, path to npz file containing, or dictionary containing FID statistics
    -- model        : Instance of inception model
    -- batch_size   : Batch size
    -- dims         : Dimensionality of activations returned by Inception
    -- device       : Device to run calculations
    -- num_workers  : Number of parallel dataloader workers
    """

    # if path_or_data is a npz file, it should contain mu and sigma
    if isinstance(path_or_data, str) and path_or_data.endswith('.npz'):
        with np.load(path_or_data) as f:
            m, s = f['mu'][:], f['sigma'][:]

    # if path_or_data is a dict, it should contain mu and sigma
    elif isinstance(path_or_data, dict):
        m, s = path_or_data['mu'][:], path_or_data['sigma'][:]

    # if path_or_data is a dataset of images, need to compute mu and sigma
    else:
        m, s = calculate_activation_statistics(path_or_data, model, batch_size, dims, device, num_workers)

    return m, s


def calculate_fid(path_or_datasets, batch_size, device, dims, num_workers=1):
    """
    Calculates the FID of two datasets based on their FID statistics

    Params:
    -- path_or_datasets : List of 2 image datasets or list of 2 paths to npz files containing FID statistics
    -- batch_size       : Batch size
    -- device           : Device to run calculations
    -- dims             : Dimensionality of activations returned by Inception
    -- num_workers      : Number of parallel dataloader workers
    """

    # load inception model with the correct block index for the needed activations
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # get FID statistics for both datasets
    m1, s1 = compute_statistics_of_path(path_or_datasets[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(path_or_datasets[1], model, batch_size,
                                        dims, device, num_workers)

    # calculate FID score between the two datasets
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_fid_stats(dataset, output_dir, batch_size, device, dims, num_workers=1):
    """
    Computes and saves the FID statistics for a dataset to a .npz file

    Params:
    -- dataset     : Image dataset for which to compute FID scores
    -- output_dir  : Path to save the FID statistics
    -- batch_size  : Batch size
    -- device      : Device to run calculations
    -- dims        : Dimensionality of activations returned by Inception
    -- num_workers : Number of parallel dataloader workers
    """

    # load inception model with the correct block index for the needed activations
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # compute and save FID statistics
    print(f"Saving FID statistics")
    m1, s1 = compute_statistics_of_path(dataset, model, batch_size, dims, device, num_workers)
    np.savez_compressed(output_dir, mu=m1, sigma=s1)


def save_fid_stats_as_dict(dataset, batch_size, device, dims, num_workers=1):
    """
    Returns the FID statistics for a dataset as a dictionary with keys 'mu' and 'sigma'

    Params:
    -- dataset     : Image dataset for which to compute FID statistics
    -- batch_size  : Batch size
    -- device      : Device to run calculations
    -- dims        : Dimensionality of activations returned by Inception
    -- num_workers : Number of parallel dataloader workers
    """

    # load inception model with the correct block index for the needed activations
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # compute and return FID statistics
    print(f"Saving FID statistics")
    m1, s1 = compute_statistics_of_path(dataset, model, batch_size, dims, device, num_workers)
    return {'mu': m1, 'sigma': s1}


#################################################################
#################################################################

# above-mentioned code has been modified to TreeVAE
# below code is new and has been added to the original file

def get_precomputed_fid_scores_path(dataset, data_name, subset, batch_size=50, device='cpu', dims=2048):
    """
    If precomputed FID statistics exist, return the path to the file.
    If not, compute the FID scores and save them to a file.

    Params:
    -- dataset     : Image dataset for which to compute FID statistics
    -- data_name   : Name of the dataset (e.g. 'mnist', 'fmnist', 'cifar10', 'celeba')
    -- subset      : Name of the subset (e.g. 'train', 'test')
    -- batch_size  : Batch size
    -- device      : Device to run calculations
    -- dims        : Dimensionality of activations returned by Inception
    """

    assert data_name in ['mnist', 'fmnist', 'cifar10', 'celeba', 'cubicc']

    # change data_name to match the name used in the precomputed stats for MNIST and FashionMNIST
    if data_name == 'mnist':
        data_name = 'MNIST'
    elif data_name == 'fmnist':
        data_name = 'FashionMNIST'

    # current path
    project_path = os.getcwd() + '/'

    # check if precomputed stats exist
    if os.path.exists(project_path + f'FID/fid_stats_precomputed/fid_stats_{data_name}_{subset}.npz'):
        # save path
        fid_stats_data_path = project_path + f'FID/fid_stats_precomputed/fid_stats_{data_name}_{subset}.npz'

    # if not, compute FID stats and save them
    else:
        fid_stats_dir = f'FID/fid_stats_precomputed/fid_stats_{data_name}_{subset}'
        save_fid_stats(dataset, fid_stats_dir, batch_size=batch_size, device=device, dims=2048)
        fid_stats_data_path = fid_stats_dir + '.npz'

    # return path to precomputed FID stats
    return fid_stats_data_path


