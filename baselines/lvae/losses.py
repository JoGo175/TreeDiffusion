"""
Loss functions for the reconstruction term of the ELBO.
"""
import torch
import torch.nn.functional as F

def loss_reconstruction_binary(x, x_decoded_mean):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = torch.flatten(x_decoded_mean, start_dim=1)
    loss = F.binary_cross_entropy(input = x_decoded_mean, target = x, reduction='none').sum(dim=-1)
    return loss

def loss_reconstruction_mse(x, x_decoded_mean):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = torch.flatten(x_decoded_mean, start_dim=1)
    loss = F.mse_loss(input = x_decoded_mean, target = x, reduction='none').sum(dim=-1)
    return loss
