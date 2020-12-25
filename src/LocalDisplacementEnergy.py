# -*- coding: utf-8 -*-
"""
Deformation regularization module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import torch
import torch.nn as nn


class LocalDisplacementEnergy(nn.Module):

    def __init__(self, dimension, **kwargs):
        super(LocalDisplacementEnergy, self).__init__()
        self.dimension = dimension
        self.kwargs = kwargs

    def _gradient_dx(self, fv):
        """

        :param fv: one-direction flow of shape [..., *vol_shape]
        :return:
        """
        if self.dimension == 3:
            return (fv[..., 2:, 1:-1, 1:-1] - fv[..., :-2, 1:-1, 1:-1]) / 2
        elif self.dimension == 2:
            return (fv[..., 2:, 1:-1] - fv[..., :-2, 1:-1]) / 2
        else:
            raise NotImplementedError

    def _gradient_dy(self, fv):
        """

        :param fv: one-direction flow of shape [..., *vol_shape]
        :return:
        """
        if self.dimension == 3:
            return (fv[..., 1:-1, 2:, 1:-1] - fv[..., 1:-1, :-2, 1:-1]) / 2
        elif self.dimension == 2:
            return (fv[..., 1:-1, 2:] - fv[..., 1:-1, :-2]) / 2
        else:
            raise NotImplementedError

    def _gradient_dz(self, fv):
        """

        :param fv: one-direction flow of shape [..., *vol_shape]
        :return:
        """
        if self.dimension == 3:
            return (fv[..., 1:-1, 1:-1, 2:] - fv[..., 1:-1, 1:-1, :-2]) / 2
        else:
            raise NotImplementedError

    def _gradient_txyz(self, Txyz, fn):
        """

        :param Txyz: flow of shape [..., 3, *vol_shape]
        :param fn: gradient function
        :return: tensor of shape [..., dimension, *vol_shape]
        """
        if self.dimension == 3:
            return torch.stack([fn(Txyz[..., i, :, :, :]) for i in range(Txyz.size(-4))], dim=-4)
        elif self.dimension == 2:
            return torch.stack([fn(Txyz[..., i, :, :]) for i in range(Txyz.size(-3))], dim=-3)
        else:
            raise NotImplementedError


class BendingEnergy(LocalDisplacementEnergy):
    def __init__(self, alpha=1, **kwargs):
        super(BendingEnergy, self).__init__(**kwargs)
        self.alpha = alpha

    def forward(self, flow):
        dfdx = self._gradient_txyz(flow, self._gradient_dx)
        dfdy = self._gradient_txyz(flow, self._gradient_dy)

        dfdxx = self._gradient_txyz(dfdx, self._gradient_dx)
        dfdyy = self._gradient_txyz(dfdy, self._gradient_dy)
        dfdxy = self._gradient_txyz(dfdx, self._gradient_dy)

        if self.dimension == 2:
            return self.alpha * torch.mean(dfdxx ** 2 + dfdyy ** 2 + 2 * dfdxy ** 2)

        elif self.dimension == 3:
            dfdz = self._gradient_txyz(flow, self._gradient_dz)
            dfdzz = self._gradient_txyz(dfdz, self._gradient_dz)
            dfdyz = self._gradient_txyz(dfdy, self._gradient_dz)
            dfdxz = self._gradient_txyz(dfdx, self._gradient_dz)

            return self.alpha * torch.mean(
                dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2 + 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    bending_energy = BendingEnergy(dimension=2)

    ddf = torch.rand(1, 5, 2, 224, 224)
    energy = bending_energy(ddf)

    print(energy.item())
    # print(bending_energy.dimension)
