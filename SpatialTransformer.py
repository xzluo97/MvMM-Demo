# -*- coding: utf-8 -*-
"""
Spatial transformer module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from AffineGrid import AffineGrid


class SpatialTransformer(nn.Module):
    """
    Spatial transformer module adapted from https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py.

    """
    def __init__(self, size, mode='bilinear'):
        """
        Instantiate the block

        :param size: size of input to the spatial transformer block
        :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        self.size = size
        self.dimension = len(self.size)

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.to(torch.float32)
        self.register_buffer('grid', grid)

        self.mode = mode

        self.affine2mesh = AffineGrid(size=self.size)

    def forward(self, src, flows=None, theta=None, mode=None, **kwargs):
        """
        Push the src and flow through the spatial transform block
        :param src: the original moving/source, image of shape [batch, channels, *vol_shape]
        :param flows: a tensor or list of tensors, each of shape [batch, dimension, *vol_shape]
        :param theta: affine matrix for affine transformation, of shape [batch, dimension, dimension+1]
        :param mode: interpolation mode
        """
        shape = src.shape[2:]
        assert len(shape) == self.dimension
        assert list(shape) == list(self.size)
        mode = mode if mode is not None else self.mode

        # if flows is None and theta is None:
        #     return src
        # print(flows)

        if theta is not None:
            affine_grid = self.affine2mesh(theta)  # [batch, dimension, *vol_shape]
            new_locs = affine_grid
        else:
            # clone the original grid, IMPORTANT NOT to alter the original grid
            new_locs = self.grid.clone().repeat(src.size(0), *[1]*(self.dimension + 1))  # [batch, dimension, *vol_shape]

        if flows is not None:
            if isinstance(flows, torch.Tensor):
                new_locs = new_locs + flows
            elif isinstance(flows, (list, tuple)):
                compose_type = kwargs.pop('compose_type', 'additive')
                if compose_type == 'additive':
                    for flow in flows:
                        if flow is not None:
                            new_locs = new_locs + flow
            else:
                raise NotImplementedError

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(self.dimension):
            new_locs[:, i] = 2 * (new_locs[:, i] / (self.size[i] - 1) - 0.5)

        if self.dimension == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            # if kwargs.pop('permute_channels', True):
            new_locs = new_locs[..., [1,0]]
        elif self.dimension == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)  # change shape to [batch, *vol_shape, dimension]
            # if kwargs.pop('permute_channels', True):
            # permute channels into [D, H, W], which is necessary for F.grid_sample
            new_locs = new_locs[..., [2,1,0]]
        else:
            raise NotImplementedError

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode, padding_mode='border')


if __name__ == '__main__':
    transform = SpatialTransformer(size=(224, 224))

    x = torch.randn(1, 4, 224, 224)
    t1 = torch.zeros(1, 2, 224, 224)
    t2 = torch.zeros(1, 2, 224, 224)
    t3 = torch.zeros(1, 2, 224, 224)

    # y = transform(x, [t1, t2, t3])
    y = transform(x, None)
    y1 = transform(x, t1)
    print(nn.L1Loss()(x, y1).mean())

    # print(x)
    # print(y)

    # affine = AffineGrid(size=(20, 30))
    # theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32).repeat(10, 1, 1)
    # grid = affine(theta)
    # print(grid.size())
    # print(grid.min(), grid.max())

