# -*- coding: utf-8 -*-
"""
Modules for computing metrics.

@author: Xinzhe Luo
"""

import numpy as np
import torch
import torch.nn as nn


##########################################################################
# Helper functions
#########################################################################3
def get_segmentation(predictor, mode='torch'):
    """
    produce the segmentation maps from the probability maps
    """
    assert mode in ['torch', 'np'], "The mode must be either 'torch' or 'np'!"
    if mode == 'torch':
        assert isinstance(predictor, torch.Tensor)
        return torch.where(torch.eq(torch.max(predictor, dim=1, keepdim=True)[0], predictor),
                           torch.ones_like(predictor),
                           torch.zeros_like(predictor))

    elif mode == 'np':
        assert isinstance(predictor, np.ndarray)
        return np.where(np.equal(np.max(predictor, 1, keepdims=True), predictor),
                        np.ones_like(predictor),
                        np.zeros_like(predictor))



##########################################################################
# Hyper classes for metrics construction.
##########################################################################
class OverlapMetrics(nn.Module):
    """
    Compute the Dice similarity coefficient between the ground truth and the prediction.
    Assume the first class is background.

    """
    def __init__(self, eps=1e-5, mode='torch', type='average_foreground_dice', **kwargs):
        super(OverlapMetrics, self).__init__()
        self.eps = eps
        self.mode = mode
        self.type = type
        self.kwargs = kwargs
        self.class_index = kwargs.get('class_index', None)
        self.channel_last = kwargs.get('channel_last', False)

        assert mode in ['torch', 'np'], "The mode must be either 'tf' or 'np'!"
        assert type in ['average_foreground_dice', 'class_specific_dice', 'average_foreground_jaccard']

    def forward(self, y_true, y_seg):
        """

        :param y_true: tensor of shape [batch, num_classes, *vol_shape]
        :param y_seg: tensor of shape [batch, num_classes, *vol_shape]
        :return:
        """
        if self.mode == 'np':
            y_true = torch.from_numpy(y_true)
            y_seg = torch.from_numpy(y_seg)

        if self.channel_last:
            dimension = y_true.dim() - 2
            y_true = y_true.permute(0, -1, *list(range(1, 1 + dimension)))
            y_seg = y_seg.permute(0, -1, *list(range(1, 1 + dimension)))

        assert y_true.size()[1:] == y_seg.size()[1:], "The ground truth and prediction must be of equal shape! " \
                                                      "Ground truth shape: %s, " \
                                                      "prediction shape: %s" % (tuple(y_true.size()),
                                                                                tuple(y_seg.size()))

        n_class = y_seg.size()[1]

        y_seg = get_segmentation(y_seg, mode='torch')

        if self.type == 'average_foreground_dice':
            dice = 0.
            for i in range(1, n_class):
                top = 2 * torch.sum(y_true[:, i] * y_seg[:, i])
                bottom = torch.sum(y_true[:, i] + y_seg[:, i])
                dice += top / (bottom + self.eps)

            metric = torch.div(dice, torch.tensor(n_class - 1, dtype=torch.float32))

        elif self.type == 'class_specific_dice':
            assert self.class_index is not None, "The class index must be provided!"
            top = 2 * torch.sum(y_true[:, self.class_index] * y_seg[:, self.class_index])
            bottom = torch.sum(y_true[:, self.class_index] + y_seg[:, self.class_index])
            metric = top / (bottom + self.eps)

        elif self.type == 'average_foreground_jaccard':
            jaccard = 0.
            y_true = y_true.type(torch.bool)
            y_seg = y_seg.type(torch.bool)
            for i in range(1, n_class):
                top = torch.sum(y_true[:, i] & y_seg[:, i], dtype=torch.float32)
                bottom = torch.sum(y_true[:, i] | y_seg[:, i], dtype=torch.float32)
                jaccard += top / (bottom + self.eps)

            metric = torch.div(jaccard, torch.tensor(n_class - 1, dtype=torch.float32))

        else:
            raise ValueError("Unknown overlap metric: %s" % self.type)

        if self.mode == 'np':
            return metric.detach().numpy()

        return metric

