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


if __name__ == '__main__':
    # load data
    data_path = '../MvMM_demo'

    os.chdir(data_path)
    print(os.getcwd())

    # load data
    # image_names = ['patient1_DE_image_slice12.nii.gz',
    #                'patient1_DE_image_slice12_aug1.nii.gz',
    #                'patient1_DE_image_slice12_aug2.nii.gz']

    image_names = ['patient1_DE_image_slice12.nii.gz',
                   'patient1_C0_image_slice5.nii.gz',
                   'patient1_T2_image_slice3.nii.gz']
    images = [image_utils.load_image_nii(name)[0] for name in image_names]  # [224, 224, 1]
    labels = [image_utils.load_image_nii(name.replace('image', 'label'))[0] for name in image_names]

    atlas_label = image_utils.load_image_nii('patient1_C0_label_slice5_affine.nii.gz')[0]
    original_atlas = torch.from_numpy(atlas_label).unsqueeze(0).unsqueeze(0).squeeze(-1)

    # preprocess data
    images = [image.squeeze(2) for image in images]
    original_images = torch.stack([torch.from_numpy(image) for image in images]).unsqueeze(1).unsqueeze(0)  # [1, 3, 1, 224, 224]
    labels = [label.squeeze(2) for label in labels]
    original_labels = torch.stack([torch.from_numpy(label) for label in labels]).unsqueeze(1).unsqueeze(0)
    label_intensities = (0, 200, 500, 600)
    images = [np.clip(image, np.percentile(image, 1), np.percentile(image, 99)) for image in images]
    images = [image_utils.normalize_image(image, normalization='min-max') for image in images]
    labels = [image_utils.get_one_hot_label(label, label_intensities, channel_first=True) for label in labels]
    atlas_label = image_utils.get_one_hot_label(atlas_label.squeeze(2), label_intensities, channel_first=True)


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
    model = MvMMVEM(vol_shape=(224, 224), num_subjects=3,
                    num_classes=4, num_subtypes=(2, 2, 2, 2),
                    flow_scales=(0, 1, 2), mask_sigma=3, transform='affine')

    # initialize parameters
    model.init_parameters(images, prior=prior)


    # set hyper-parameters
    training_iters = 5000
    EM_steps = 3
    display_steps = 20
    bending_energy = 10

    # set optimizer
    # optimizer for rigid transformation
    # optimizer = torch.optim.Adam(params=[{'params': model.rotate_params.parameters(), 'lr': 0.0003},
    #                                      {'params': model.transl_params.parameters(), 'lr': 0.0003},
    #                                      {'params': model.vectors.parameters(), 'lr': 0.001}])

    # optimizer for affine transformation
    optimizer = torch.optim.Adam(params=[{'params': model.theta.parameters(), 'lr': 0.0003},
                                         {'params': model.vectors.parameters(), 'lr': 0.001}])

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
