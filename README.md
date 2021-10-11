# Multivariate mixture model in PyTorch

>This is a course project in *Medical Image Analysis* on “Multivariate mixture model for myocardial segmentation combining multi-source images”. The algorithm is re-implemented in PyTorch. The lecture notes for illustration and presentation is also included.

## Getting Started

### Project structure

The project contains PyTorch implementation of the algorithm from Multivariate mixture model on 2D multi-sequence cardiac MR images. The data can be downloaded from [MS-CMRSeg-2019 challenge](https://zmiclab.github.io/projects/mscmrseg19/). The project structure is as follows:

```
MvMM-Demo
|-- src
|   |-- AffineGrid.py                # convert affine matrix to resampling grid
|   |-- LocalDisplacementEnergy.py   # displacement regularization, bending energy
|   |-- MvMMVEM.py                   # model construction and EM algorithm
|   |-- MvMMVEMDemo.py               # Demo: image loading, preprocessing, model optimization and result visualization
|   |-- SpatialTransformer.py        # spatial transformation module
|   |-- image_utils.py               # functions for image loading and preprocessing
|   |-- metrics.py                   # metrics computation
|   |-- utils.py                     # utility functions
```

### Usage

Combined segmentation from a set of images is achieved by:

```
python MvMMVEMDemo.py 
--data_path #YOUR OWN DATA PATH#           # data path to load images
--image_names #YOUR OWN IMAGE NAMES#       # image names
--atlas_name #YOUR OWN ARLAS NAME#         # atlas name
--label_intensities 0 255                  # label intensity values
--vol_shape 256 256                        # image size
--num_subjects 3                           # number of subjects
--num_classes 2                            # number of classes
--num_subtypes 2 2                         # number of subtypes
--transform rigid                          # transformation type
--training_iters 1000                      # training iterations
--EM_steps 3                               # EM update steps
--bending_energy 1                         # bending energy coefficient
```

## Citation

If you found the project useful, please cite our papers as below:

```
@misc{Luo2020MultivariateMixtureModel,
  title={Medical Image Analysis, Multivariate Mixture Model for Combined Computing},
  author={Xinzhe Luo},
  year={2020}
}
```

## Contact

For any questions or problems please [open an issue](https://github.com/xzluo97/MvMM-Demo/issues/new) on GitHub.