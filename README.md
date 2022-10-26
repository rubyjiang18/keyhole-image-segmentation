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

├── images<br/>
│   └── *.jpg  # keyhole images for segmentation<br/>  
│── masks<br/>
│   └── *.png  # segmentation masks corresponds to the images<br/>
└── segmentation<br/>
    │── train.txt  # image list for training<br/>
    │── trainval.txt  # all images<br/>
    |── val.txt  # image list for validation<br/>
    └── test.txt  # image list for test<br/>

.
├── build                   # Compiled files (alternatively `dist`)
├── docs                    # Documentation files (alternatively `doc`)
├── src                     # Source files (alternatively `lib` or `app`)
├── test                    # Automated tests (alternatively `spec` or `tests`)
├── tools                   # Tools and utilities
├── LICENSE
└── README.md
