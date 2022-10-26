# keyhole-image-segmentation
This is the repo for keyhole image segmentation


## Keyhole Segmentation Dataset
## Data sources
The original data come from 32ID APS at Argonne National Laboratory
## Specification
The dataset contains xxx for training, xxx images for validation and xxx for test
The files in `segmentation` directory contains the image list.
## Directory Structure
keyhole_segmentation

├── images
│   └── *.jpg  # images for segmentation
│── masks
│   └── *.png  # segmentation masks corresponds to the images
└── segmentation
    │── train.txt  # image list for training
    │── trainval.txt  # all images
    └── val.txt  # image list for validation
