# -*- coding: utf-8 -*-
"""
Data utility functions for image loader and processing.

@author: Xinzhe Luo
@version: 0.1
"""


from __future__ import print_function, division, absolute_import, unicode_literals

import math
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import rescale
from scipy import stats, signal
import matplotlib.pyplot as plt
from utils import get_normalized_prob


def load_image_nii(path, dtype=np.float32, scale=0, order=1):
    img = nib.load(path)
    image = np.asarray(img.get_fdata(), dtype)
    if scale > 0:
        image = rescale(image, 1 / (2 ** scale), mode='reflect',
                        multichannel=False, anti_aliasing=False, order=order)
    return image, img.affine, img.header


def save_image_nii(array, save_path, **kwargs):
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_dtype = kwargs.pop("save_dtype", np.uint16)
    img = nib.Nifti1Image(np.asarray(array, dtype=save_dtype), affine=affine, header=header)
    nib.save(img, save_path)


def normalize_image(image, normalization=None, **kwargs):
    if normalization == 'min-max':
        image -= np.min(image)
        image /= np.max(image)

    elif normalization == 'z-score':
        image = stats.zscore(image, axis=None, ddof=1)
        if kwargs.pop('clip_value', None):
            image = np.clip(image, -3, 3)

    elif normalization == 'interval':
        image -= np.min(image)
        image /= np.max(image)
        a = kwargs.pop('a', -1)
        b = kwargs.pop('b', 1)
        image = (b-a) * image + a

    return image


def get_one_hot_label(gt, label_intensities, channel_first=False):
    """
    Process label data into one-hot representation.

    :param gt: A ground-truth array, of shape [*vol_shape].
    :return: An array of one-hot representation, of shape [num_classes, *vol_shape].
    """
    num_classes = len(label_intensities)
    label = np.around(gt)
    if channel_first:
        label = np.zeros((np.hstack((num_classes, label.shape))), dtype=np.float32)

        for k in range(1, num_classes):
            label[k] = (gt == label_intensities[k])

        label[0] = np.logical_not(np.sum(label[1:,], axis=0))
    else:
        label = np.zeros((np.hstack((label.shape, num_classes))), dtype=np.float32)

        for k in range(1, num_classes):
            label[..., k] = (gt == label_intensities[k])

        label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))

    return label


def visualize_image2d(image, **kwargs):
    """

    :param image: a image of shape [nx, ny, channel]
    :return:
    """
    plt.imshow(image, **kwargs)
    plt.show()


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = np.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / np.sum(k)


def separable_filter2d(vol, kernel, mode='torch'):
    """
    3D convolution using separable filter along each axis

    :param vol: torch tensor of shape [batch, channels, *vol_shape]
    :param kernel: of shape [k]
    :return: of shape [batch, channels, *vol_shape]
    """
    if np.all(kernel == 0):
        return vol
    if mode == 'torch':
        kernel = torch.tensor(kernel, dtype=torch.float32).to(vol.device)
        channels = vol.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv2d(F.conv2d(vol, kernel.view(channels, 1, -1, 1), padding=(padding, 0), groups=channels),
                        kernel.view(channels, 1, 1, -1), padding=(0, padding), groups=channels)
    elif mode == 'np':
        return signal.convolve(signal.convolve(vol, np.reshape(kernel, [1, 1, -1, 1]), 'same'),
                                               np.reshape(kernel, [1, 1, 1, -1]), 'same')

def get_prob_from_label(label, dimension=2, sigma=1., mode='torch', **kwargs):
    """
    Produce probability map from one-hot labels.

    :param label: one hot label of shape [n_batch, num_classes, *vol_shape]
    :param sigma: isotropic standard deviation of the Gaussian filter
    :param mode: 'torch' or 'np'
    :param kwargs:
    :return:
    """

    eps = kwargs.pop('eps', math.exp(-3**2/2))
    if dimension == 2:
        spatial_filter = separable_filter2d
    else:
        raise NotImplementedError

    if mode == 'torch':
        blur = spatial_filter(label, gauss_kernel1d(sigma))
        prob = get_normalized_prob(blur, eps=eps)
    elif mode == 'np':
        blur = spatial_filter(label, gauss_kernel1d(sigma), mode='np')
        prob = get_normalized_prob(blur, mode='np', eps=eps)
    else:
        raise NotImplementedError
    return prob
