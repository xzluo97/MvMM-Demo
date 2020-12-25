# -*- coding: utf-8 -*-
"""
Multivariate mixture model with Expectation-Maximization optimization.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import sys
sys.path.append('..')
import torch.nn as nn
import torch
import torch.nn.functional as F
import SpatialTransformer, LocalDisplacementEnergy
import image_utils
import utils
# import math


class MvMMVEM(nn.Module):
    """
    Construct the model

    """
    dimension = 2

    def __init__(self, vol_shape, num_subjects, num_classes=2, num_subtypes=(1, 1), int_steps=0, eps=1e-5, **kwargs):
        super(MvMMVEM, self).__init__()
        self.vol_shape = vol_shape
        # self.dimension = len(self.vol_shape)
        self.num_subjects = num_subjects
        self.num_classes = num_classes
        self.num_subtypes = num_subtypes
        assert len(self.num_subtypes) == self.num_classes
        self.int_steps = int_steps
        self.eps = eps
        self.kwargs = kwargs
        self.spatial_sigma = self.kwargs.pop('spatial_sigma', 1)
        self.mask_sigma = self.kwargs.pop('mask_sigma', 1)
        self.transform = self.kwargs.pop('transform', 'rigid')
        self.flow_scales = sorted(self.kwargs.pop('flow_scales', (0, )), reverse=True)  # sort input

        # spatial transformer
        self.spatial_transform = SpatialTransformer.SpatialTransformer(self.vol_shape)

        # registration parameters of the probabilistic atlas
        self.vectors = nn.ParameterDict(dict(zip(['scale_%s' % s for s in self.flow_scales],
                                                 [nn.Parameter(torch.zeros(1, self.dimension,
                                                                           *[v // 2 ** s for v in self.vol_shape]))
                                                  for s in self.flow_scales])))

        if self.transform == 'affine':
            if self.dimension == 2:
                self.theta = nn.ParameterList([nn.Parameter(torch.tensor([[[1, 0, 0], [0, 1, 0]]],
                                                                         dtype=torch.float32))
                                               for _ in range(self.num_subjects - 1)])  # [1, 2, 3]

        elif self.transform == 'rigid':
            if self.dimension == 2:
                self.rotate_params = nn.ParameterList([nn.Parameter(torch.tensor([0], dtype=torch.float32))
                                                       for _ in range(self.num_subjects - 1)])
                self.transl_params = nn.ParameterList([nn.Parameter(torch.tensor([[0], [0]], dtype=torch.float32))
                                                       for _ in range(self.num_subjects - 1)])
        else:
            raise NotImplementedError

    def init_parameters(self, images, prior, device='cpu'):
        """

        :param images: tensor of shape [1, num_subjects, 1, *vol_shape]
        :param prior: tensor of shape [1, num_classes, *vol_shape]
        :param device:
        :return:
        """
        self.mask = self._spatial_filter(prior[:, 1:].sum(dim=1, keepdim=True),
                                         image_utils.gauss_kernel1d(self.mask_sigma)).gt(self.eps).to(torch.float32)

        self.pi = torch.sum(prior * self.mask, dim=list(range(2, 2 + self.dimension)),
                            keepdim=True) / torch.sum(prior * self.mask, dim=list(range(1, 2 + self.dimension)), keepdim=True)
        self.prior = utils.compute_normalized_prob(self.pi * prior, dim=1)

        self.tau = [torch.full((self.num_subjects, self.num_subtypes[i]), 1 / self.num_subtypes[i])
                    for i in range(self.num_classes)]

        mu_k = torch.sum(images * prior.unsqueeze(1) * self.mask,
                         dim=list(range(3, 3 + self.dimension))) / torch.sum(prior.unsqueeze(1) * self.mask,
                                                                             dim=list(range(3, 3 + self.dimension))).clamp(min=self.eps)  # [1, num_subjects, num_classes]
        sigma2_k = torch.sum((images - mu_k.view(1, -1, self.num_classes, *[1] * self.dimension)) ** 2 * prior.unsqueeze(1) * self.mask,
                             dim=list(range(3, 3 + self.dimension))) / torch.sum(prior.unsqueeze(1) * self.mask,
                                                                                 dim=list(range(3, 3 + self.dimension))).clamp(min=self.eps)  # [1, num_subjects, num_classes]

        self.mu = []
        for i in range(self.num_classes):
            if self.num_subtypes[i] == 1:
                self.mu.append(mu_k[:, :, [i]].squeeze(0))  # [num_subjects, 1]
            else:
                a = torch.linspace(-1, 1, steps=self.num_subtypes[i])  # [num_subtypes[i]]
                self.mu.append(mu_k[:, :, [i]].squeeze(0) + a.unsqueeze(0) * sigma2_k[:, :, [i]].squeeze(0).sqrt())  # [num_subjects, num_subtypes[i]]

        self.sigma2 = [sigma2_k[:, :, [i]].squeeze(0).mul(self.num_subtypes[i]).repeat(1, self.num_subtypes[i])
                       for i in range(self.num_classes)]

        self.posterior = self.prior

        # to device
        self.pi = self.pi.to(device)
        self.prior = self.prior.to(device)
        self.posterior = self.posterior.to(device)
        for i in range(self.num_classes):
            self.tau[i] = self.tau[i].to(device)
            self.mu[i] = self.mu[i].to(device)
            self.sigma2[i] = self.sigma2[i].to(device)

        # # crop mask as the dilated foreground of posterior
        # self.mask = self._compute_mask_from_posterior(self.posterior)

    def _spatial_filter(self, *args, **kwargs):
        # spatial smoothing
        if self.dimension == 2:
            spatial_filter = image_utils.separable_filter2d
        else:
            raise NotImplementedError

        return spatial_filter(*args, **kwargs)

    def _compute_mask_from_posterior(self, posterior):
        """

        :param posterior: tensor of shape [1, num_classes, *vol_shape]
        :return:
        """
        # crop mask as the dilated foreground of posterior
        foreground = 1 - posterior[:, 0].unsqueeze(1)
        dilated_foreground = self._spatial_filter(foreground, image_utils.gauss_kernel1d(self.mask_sigma))
        return (dilated_foreground > self.eps).to(torch.float32)  # [1, 1, *vol_shape]

    def forward(self, images, prior, **kwargs):
        """

        :param images: tensor of shape [1, num_subjects, img_channels, *vol_shape]
        :param prior: tensor of shape [1, num_classes, *vol_shape]
        :return:
        """
        flow_scale = sorted(kwargs.pop('flow_scale', (0,)), reverse=True)
        # freeze gradient
        for s in self.flow_scales:
            if s in flow_scale:
                self.vectors['scale_%s' % s].requires_grad = True
            else:
                self.vectors['scale_%s' % s].requires_grad = False

        if self.dimension == 2:
            upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # registration
        self.flows = [F.upsample(self.vectors['scale_%s' % s], scale_factor=2 ** s, mode=upsample_mode,
                                 align_corners=True) for s in self.flow_scales]
        if self.transform == 'rigid':
            self.theta = [torch.stack([torch.cat([torch.cos(self.rotate_params[i]),
                                                  - torch.sin(self.rotate_params[i]),
                                                  self.transl_params[i][0]]),
                                       torch.cat([torch.sin(self.rotate_params[i]),
                                                  torch.cos(self.rotate_params[i]),
                                                  self.transl_params[i][1]])]).unsqueeze(0)
                          for i in range(self.num_subjects - 1)]

        reg_mode = kwargs.pop('reg_mode', None)
        warped_prior = self.transform_prior(prior)
        self.warped_mask = self.transform_prior(self.mask, mode='nearest').detach()
        warped_prior = utils.compute_normalized_prob(warped_prior * self.pi)
        warped_images = torch.stack(self.transform_images(images, mode=reg_mode), dim=1)

        return warped_images, warped_prior

    def transform_prior(self, prior, **kwargs):
        mode = kwargs.pop('mode', None)
        warped_prior = self.spatial_transform(prior, flows=self.flows, mode=mode, compose_type='additive')
        return warped_prior

    def transform_images(self, images, **kwargs):
        """

        :param images: tensor of shape [1, num_subjects, img_channels, *vol_shape]
        :return: list of warped images, each of shape [1, img_channels, *vol_shape]
        """
        mode = kwargs.pop('mode', None)
        warped_images = [self.spatial_transform(images[:, i], theta=self.theta[i - 1] if (i > 0) else None,
                                                mode=mode) for i in range(self.num_subjects)]
        return warped_images

    def transform_labels(self, labels):
        """

        :param labels: tensor of shape [1, num_subjects, num_classes, *vol_shape]
        :return: list of warped labels, each of shape [1, num_classes, *vol_shape]
        """
        warped_labels = [self.spatial_transform(labels[:, i], theta=self.theta[i - 1] if (i > 0) else None,
                                                mode='nearest') for i in range(self.num_subjects)]
        return warped_labels

    def _compute_class_cpd(self, tau, subtype_cpds):
        """

        :param tau: list of tensors, each of shape [num_subtypes[i]]
        :param subtype_cpds: list of tensors, num_classes * [1, num_subtypes[i], *vol_shape]
        :return: class pdf of shape [1, num_classes, *vol_shape]
        """
        class_cpd = []
        for i in range(self.num_classes):
            tau_ = tau[i].view(1, self.num_subtypes[i], *[1]*self.dimension)
            class_cpd.append(torch.sum(tau_ * subtype_cpds[i], dim=1, keepdim=True))
        cpd = torch.cat(class_cpd, dim=1)
        # print(class_pdfs[0].size(), pdf.size())
        return cpd

    def _compute_subtype_cpds(self, image, mu, sigma2):
        """

        :param image: tensor of shape [1, 1, *vol_shape]
        :param mu: list of tensors, each of shape [num_subtypes[i]]
        :param sigma2: list of tensors, each of shape [num_subtypes[i]]
        :return: list of subtype pdfs, each of shape [1, num_subtypes[i], *vol_shape]
        """
        subtype_cpds = []
        for i in range(self.num_classes):
            mu_ = mu[i].view(1, self.num_subtypes[i], *[1]*self.dimension)
            sigma_ = sigma2[i].view(1, self.num_subtypes[i], *[1]*self.dimension).sqrt()
            subtype_cpds.append(utils.gaussian_pdf(image, mu_, sigma_).to(torch.float32))
        return subtype_cpds

    def compute_subtype_class_cpds(self, images):
        subtype_cpds = [self._compute_subtype_cpds(images[:, i],
                                                   [v[i] for v in self.mu],
                                                   [v[i] for v in self.sigma2]) for i in range(self.num_subjects)]

        class_cpds = [self._compute_class_cpd([v[i] for v in self.tau], subtype_cpds[i]) for i in range(self.num_subjects)]

        return class_cpds, subtype_cpds


    def update(self, warped_images_grad, warped_prior_grad):
        """
        update appearance parameters

        :param warped_images: tensor of shape [1, num_subjects, 1, *vol_shape]
        :param warped_prior: tensor of shape [1, num_classes, *vol_shape]
        :return:
        """

        # E-step: update the posterior
        class_cpds_grad, subtype_cpds_grad = self.compute_subtype_class_cpds(warped_images_grad)

        # detach variables from gradient calculation
        subtype_cpds = [[cpd.detach() for cpd in subtype_cpds_grad[i]] for i in range(self.num_subjects)]
        class_cpds = [cpd.detach() for cpd in class_cpds_grad]
        warped_images = warped_images_grad.detach()
        warped_prior = warped_prior_grad.detach()

        data_likelihood = torch.stack(class_cpds, dim=1).clamp(min=self.eps).log().sum(dim=1).exp()
        self.posterior = utils.compute_normalized_prob(data_likelihood * warped_prior, dim=1)  # [1, num_classes, *vol_shape]

        # M-step: update the parameters
        # update pi, prior
        self.pi = torch.sum(self.posterior * self.warped_mask, dim=list(range(2, 2 + self.dimension)),
                            keepdim=True) / torch.sum(warped_prior / torch.sum(warped_prior * self.pi, dim=1,
                                                                               keepdim=True).clamp(min=self.eps) * self.warped_mask,
                                                      dim=list(range(2, 2 + self.dimension)), keepdim=True).clamp(min=self.eps)
        # warped_prior_grad = utils.compute_normalized_prob(self.pi * warped_prior_grad, dim=1)
        # print(self.pi.requires_grad, self.prior.requires_grad)

        # update tau, mu, sigma2
        for i in range(self.num_classes):
            tau_ = utils.compute_normalized_prob(
                self.tau[i].view(1, self.num_subjects, self.num_subtypes[i],
                                 *[1] * self.dimension) * torch.stack([cpd[i] for cpd in subtype_cpds], dim=1),
                dim=2) * self.posterior[:, [i]].unsqueeze(1)  # [1, num_subjects, num_subtypes[i], *vol_shape]
            # print(tau_.requires_grad)
            tau_ = tau_ * self.warped_mask.unsqueeze(1)

            self.tau[i] = utils.compute_normalized_prob(
                tau_.sum(dim=(0, *[i + 3 for i in range(self.dimension)])),
                dim=1)  # update tau, [num_subjects, num_subtypes[i]]
            # print(self.tau[i].requires_grad)

            self.mu[i] = torch.sum(
                tau_ * warped_images, dim=(0, *[i + 3 for i in range(self.dimension)])) / tau_.sum(
                dim=(0, *[i + 3 for i in range(self.dimension)])
            ).clamp(min=self.eps)  # update mu, [num_subjects, num_subtypes[i]]
            # print(self.mu[i].requires_grad)

            self.sigma2[i] = torch.sum(
                tau_ * (warped_images - self.mu[i].view(1, self.num_subjects, self.num_subtypes[i],
                                                        *[1] * self.dimension)
                        ) ** 2, dim=(0, *[i + 3 for i in range(self.dimension)])) / tau_.sum(
                dim=(0, *[i + 3 for i in range(self.dimension)])
            ).clamp(min=self.eps)  # update sigma square, [num_subjects, num_subtypes[i]]
            # print(self.sigma2[i].requires_grad)

        return class_cpds_grad

    def _compute_data_likelihood(self, class_cpds_grad):
        """

        :param class_cpds_grad: list of conditional probabilistic distributions, each of shape [batch, num_class, *vol_shape]
        :return: tensor of shape [batch, num_classes, *vol_shape]
        """
        data_likelihood = torch.stack(class_cpds_grad, dim=1).clamp(min=self.eps).log().sum(dim=1).exp()

        return data_likelihood

    def estimate_posterior(self, *args, **kwargs):
        # likelihood = self._compute_likelihood(*args, **kwargs)
        return self.posterior

    def loss_function(self, class_cpds_grad, warped_prior_grad, alpha=0.1, **kwargs):
        likelihood = self._compute_data_likelihood(class_cpds_grad) * warped_prior_grad

        likelihood = likelihood * self.warped_mask

        sum_likelihood = likelihood.sum(dim=1)
        log_likelihood = sum_likelihood.clamp_min(self.eps).log()

        # print(mask.sum().item())
        mask = (likelihood > self.eps).to(torch.float32)
        loss = - torch.sum(log_likelihood * mask) / torch.sum(mask).add(self.eps)
        regularization = self._compute_regularization(alpha)
        # print(loss.item(), regularization.item())
        loss += regularization

        return loss

    def _compute_regularization(self, alpha):
        regularization = 0.
        if alpha > 0:
            self.bending_energy = LocalDisplacementEnergy.BendingEnergy(alpha, dimension=self.dimension)
            # print(self.bending_energy(flows).item())
            for flow in self.flows:
                regularization += self.bending_energy(flow)

        return regularization


if __name__ == '__main__':
    model = MvMMVEM((224, 224), 3, 4, (2, 2, 1, 1), int_steps=0, )
    images = torch.rand(1, 3, 1, 224, 224)
    prior = torch.rand(1, 4, 224, 224)
    labels = torch.rand(1, 3, 4, 224, 224)


    model.init_parameters(images, prior)
    warped_images, warped_prior = model(images, prior)

    class_cpds_grad = model.update(warped_images, warped_prior)

    loss = model.loss_function(class_cpds_grad, alpha=0.1)

    print(loss.item())

    loss.backward()

    print(model.rotate_params[0].grad, model.transl_params[0].grad)

    print(model.vectors['scale_0'].grad)

    # model.reset_parameters()
