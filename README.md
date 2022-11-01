# keyhole-image-segmentation
This is the repo for keyhole image segmentation

## Keyhole Segmentation Models
1. Unet
2. Unet + ResNet 
3. Unet + MobileNet
4. Unet + ConvNeXt
5. Deeplab v3
6. Deeplab + ResNet 
7. DeepLab + MobileNet
8. DeepLab + ConvNeXt

## Keyhole Segmentation Dataset
### Data sources
The original data come from 32ID APS at Argonne National Laboratory。

| Material      | Type | Power (W)     | Scan speed (mm/s) | Number of images
| ------------- |:-------------:|:-----:| -----:|
| Al6061      | Moving Laser |right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


### Specification
The dataset contains xxx for training, xxx images for validation and xxx for test
The files in `segmentation` directory contains the image list.
### Directory Structure
keyhole_segmentation

    .
    ├── image
    │   ├── *.jpg               # keyhole images for segmentation
    ├── masks
    │   ├── *.png               # segmentation masks corresponds to the images
    ├── segmentation           
    │   ├── train.txt           # image list for training
    │   ├── trainval.txt        # all images in training and validation
    │   ├── val.txt             # image list for validation
    │   └── test.txt            # image list for test
    └── ...
