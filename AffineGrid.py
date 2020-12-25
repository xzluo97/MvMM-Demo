# -*- coding: utf-8 -*-
"""
Affine grid module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
import torch.nn as nn


class AffineGrid(nn.Module):
    """
    Convert a affine matrix to a dense sampling grid.
    """
    def __init__(self, size):
        super(AffineGrid, self).__init__()
        self.size = size
        self.dimension = len(self.size)

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        # grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.to(torch.float32)  # [dimension, *vol_shape]
        self.register_buffer('grid', grid)

    def forward(self, theta):
        """

        :param theta: affine matrix, of shape [batch, dimension, dimension+1]
        :return: a tensor of shape [batch, dimension, *vol_shape]
        """
        assert theta.size(1) == self.dimension
        mesh = self.grid.view(self.dimension, -1)  # [dimension, num_voxels]
        mesh_aug = torch.cat([mesh, torch.ones(1, mesh.size(1), dtype=torch.float32).to(mesh.device)],
                             dim=0)  # [dimension+1, num_voxels]

        affine_grid = torch.matmul(theta, mesh_aug)  # [batch, dimension, num_voxels]

        return affine_grid.view(-1, self.dimension, *self.size)
