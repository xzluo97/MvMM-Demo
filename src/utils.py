# -*- coding: utf-8 -*-
"""

@author: Xinzhe Luo
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import math
import torch


#######################################################################
# Functions for probability computation

def gaussian_pdf(x, mu, sigma, mode='torch', eps=1e-7):
    """

    :param x: image of shape [batch, channels, *vol_shape]
    :param mu:
    :param sigma:
    :param mode:
    :param eps:
    :return:
    """
    if mode == 'torch':
        # Normal = torch.distributions.normal.Normal(mu, spatial_sigma)
        # return Normal.log_prob(x).exp()
        pi = torch.tensor(math.pi, dtype=torch.float32, device=x.device)
        return torch.clamp_min(1 / (torch.sqrt(2*pi)*sigma+eps) * torch.exp(-(x-mu)**2 / (2*sigma**2+eps)), min=eps)
    elif mode == 'np':
        return np.clip(1 / (np.sqrt(2*math.pi)*sigma+eps) * np.exp(-(x-mu)**2 / (2*sigma**2+eps)), a_min=eps)
    else:
        raise NotImplementedError


def compute_normalized_prob(prob, dim=1, mode='torch', eps=1e-5):
    if mode == 'torch':
        return prob / torch.sum(prob, dim=dim, keepdim=True).clamp(min=eps)
    elif mode == 'np':
        return prob / np.sum(prob, axis=dim, keepdims=True).clip(min=eps)
    else:
        raise NotImplementedError


def get_normalized_prob(prob, mode='torch', dim=1, **kwargs):
    """
    Compute normalized probability map given the input un-normalized probabilities.

    :param prob: The un-normalized probabilities of shape [n_batch, num_classes, *vol_shape].
    :param mode: 'np' or 'torch'.
    :return: A tensor of shape [n_batch, num_classes, *vol_shape], representing the probabilities of each class.
    """
    eps = kwargs.pop('eps', 1e-7)
    if mode == 'torch':
        return prob / torch.sum(prob, dim=dim, keepdim=True).clamp(min=eps)
    elif mode == 'np':
        return prob / np.sum(prob, axis=dim, keepdims=True).clip(min=eps)
    else:
        raise NotImplementedError

