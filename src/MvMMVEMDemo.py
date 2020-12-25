# -*- coding: utf-8 -*-
"""
Unified framework for network training, validation and testing of IterativeMvMM.

__author__ = 'Xinzhe Luo'
__version__ = 0.1
"""

import sys

sys.path.append('..')
import os
import torch
import numpy as np
import image_utils
from MvMMVEM import MvMMVEM
import metrics
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MvMM-Demo')
    parser.add_argument('--data_path', default='../demo_data', type=str, help='path where to load data')
    parser.add_argument('--image_names', type=str, nargs='+', help='images to load')
    parser.add_argument('--atlas_name', type=str, help='atlas to load')
    parser.add_argument('--label_intensities', type=int, nargs='+', help='label intensities')
    parser.add_argument('--vol_shape', type=int, nargs=2, help='image size')
    parser.add_argument('--num_subjects', type=int, help='number of images included in the model')
    parser.add_argument('--num_classes', type=int, help='number of tissue types')
    parser.add_argument('--num_subtypes', type=int, nargs='+', help='number of subtypes in each tissue type')
    parser.add_argument('--transform', type=str, choices=['affine', 'rigid'], help='rigid or affine registration')
    parser.add_argument('--training_iters', type=int, help='training iterations')
    parser.add_argument('--EM_steps', type=int, help='EM steps in every iteration')
    parser.add_argument('--bending_energy', type=float, help='regualarization coefficient for bending energy')
    args = parser.parse_args()

    # load data
    data_path = args.data_path

    os.chdir(data_path)
    print(os.getcwd())

    # load data
    images = [image_utils.load_image_nii(name)[0] for name in args.image_names]  # [224, 224, 1]
    labels = [image_utils.load_image_nii(name.replace('image', 'label'))[0] for name in args.image_names]

    atlas_label = image_utils.load_image_nii(args.atlas_name)[0]
    original_atlas = torch.from_numpy(atlas_label).unsqueeze(0).unsqueeze(0).squeeze(-1)

    # preprocess data
    images = [image.squeeze(2) for image in images]
    original_images = torch.stack([torch.from_numpy(image) for image in images]).unsqueeze(1).unsqueeze(0)  # [1, 3, 1, 224, 224]
    labels = [label.squeeze(2) for label in labels]
    original_labels = torch.stack([torch.from_numpy(label) for label in labels]).unsqueeze(1).unsqueeze(0)

    images = [np.clip(image, np.percentile(image, 1), np.percentile(image, 99)) for image in images]
    images = [image_utils.normalize_image(image, normalization='min-max') for image in images]
    labels = [image_utils.get_one_hot_label(label, args.label_intensities, channel_first=True) for label in labels]
    atlas_label = image_utils.get_one_hot_label(atlas_label.squeeze(2), args.label_intensities, channel_first=True)


    fig, ax = plt.subplots(4, 3, figsize=(12, 10))

    # transfer data as input
    images = torch.from_numpy(np.stack(images)).unsqueeze(1).unsqueeze(0)  # [1, 3, 1, 224, 224]
    images = torch.stack([image_utils.separable_filter2d(images[:, i], image_utils.gauss_kernel1d(1)) for i in range(3)], dim=1)
    labels = torch.from_numpy(np.stack(labels)).unsqueeze(0)  # [1, 3, 4, 224, 224]
    atlas_label = torch.from_numpy(atlas_label).unsqueeze(0)  # [1, 4, 224, 224]
    prior = image_utils.get_prob_from_label(atlas_label, dimension=2, sigma=2)

    # set metric
    Dice = metrics.OverlapMetrics(type='average_foreground_dice')

    # instantiate model
    model = MvMMVEM(vol_shape=args.vol_shape, num_subjects=args.num_subjects,
                    num_classes=args.num_classes, num_subtypes=args.num_subtypes,
                    flow_scales=(0, 1, 2), mask_sigma=3, transform=args.transform)

    # initialize parameters
    model.init_parameters(images, prior=prior)


    # set hyper-parameters
    training_iters = args.training_iters
    EM_steps = args.EM_steps
    display_steps = 20
    bending_energy = args.bending_energy

    # set optimizer
    if args.transform == 'rigid':
        # optimizer for rigid transformation
        optimizer = torch.optim.Adam(params=[{'params': model.rotate_params.parameters(), 'lr': 0.0003},
                                             {'params': model.transl_params.parameters(), 'lr': 0.0003},
                                             {'params': model.vectors.parameters(), 'lr': 0.001}])

    elif args.transform == 'affine':
        # optimizer for affine transformation
        optimizer = torch.optim.Adam(params=[{'params': model.theta.parameters(), 'lr': 0.0003},
                                             {'params': model.vectors.parameters(), 'lr': 0.001}])
    else:
        raise NotImplementedError

    dice = []
    for i in range(model.num_subjects - 1):
        dice.append(Dice(labels[:, 0], labels[:, i + 1]).mean().item())

    print("[Original] Dice: %.4f" % (np.mean(dice)))

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(3):
        ax[0, i].imshow(np.rot90(original_images[:, i].detach().numpy().squeeze()), cmap='gray')
        ax[1, i].imshow(np.rot90(original_labels[:, i].detach().numpy().squeeze()), cmap='gray')

    ax[1, 3].imshow(np.rot90(original_atlas.detach().numpy().squeeze()), cmap='gray')
    ax[1, 3].set_title('Warped atlas')

    fig.suptitle("Original images and labels")

    plt.show()

    for step in range(training_iters):
        optimizer.zero_grad()

        # update segmentation parameters
        warped_images, warped_prior = model(images, prior=prior, flow_scale=(0, 1, 2 ))

        for j in range(EM_steps):
           _ = model.update(warped_images.detach(), warped_prior.detach())

        # update registration parameters
        class_cpds_grad, _ = model.compute_subtype_class_cpds(warped_images)

        loss = model.loss_function(class_cpds_grad, warped_prior, alpha=bending_energy)

        loss.backward()

        optimizer.step()

        if step % display_steps == (display_steps - 1):
            print(model.pi.squeeze(), model.pi.sum())
            dice = []
            warped_labels = model.transform_labels(labels)
            for i in range(model.num_subjects - 1):
                dice.append(Dice(labels[:, 0], warped_labels[i + 1]).mean().item())
            warped_atlas_label = model.transform_prior(prior=atlas_label, mode='nearest')
            atlas_dice = Dice(labels[:, 0], warped_atlas_label).mean().item()

            print("[Validation] Step: %s, Loss: %.4f, Dice: %.4f, Atlas Dice: %.4f" % (step, loss.item(),
                                                                                       np.mean(dice), atlas_dice))

            fig, ax = plt.subplots(3, 4, figsize=(12, 6))
            warped_original_images = model.transform_images(original_images)  # [1, 1, 224, 224]
            warped_original_labels = model.transform_labels(original_labels)
            warped_original_atlas = model.transform_prior(original_atlas, mode='nearest')  # []

            for i in range(3):
                ax[0, i].imshow(np.rot90(warped_original_images[i].detach().numpy().squeeze()), cmap='gray')
                ax[1, i].imshow(np.rot90(warped_images[:, i].detach().numpy().squeeze()), cmap='gray')
                ax[2, i].imshow(np.rot90(warped_original_labels[i].detach().numpy().squeeze()), cmap='gray')

            ax[0, 3].imshow(np.rot90(model.warped_mask.detach().numpy().squeeze()), cmap='gray')
            ax[0, 3].set_title('Warped mask')

            ax[1, 3].imshow(np.rot90(warped_original_atlas.detach().numpy().squeeze()), cmap='gray')
            ax[1, 3].set_title('Warped atlas')

            fig.suptitle("Transformed images and labels, Step %s" % step)

            plt.show()
