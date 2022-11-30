# keyhole-image-segmentation
This is the repo for keyhole image segmentation

## Keyhole Segmentation Models
1. Unet
2. Unet + ResNet50
3. Unet + MobileNet
5. DeepLab v3
6. Deeplab + ResNet 
7. DeepLab + MobileNet

## Relevant Github Repo
1. Unet: https://github.com/milesial/Pytorch-UNet#weights--biases
2. Segmentation models Pytorch: https://github.com/qubvel/segmentation_models.pytorch
3. [Nasa pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models)

## Keyhole Segmentation Dataset
### Data sources
The original data come from 32ID APS at Argonne National Laboratory. Images are obtained from various laser power (W) and scan speed (mm/s) with no powder layer.

Material | Type  | Number of images | Power (W) - Scan speed (m/s)
--- | --- | --- | ---
Al6061 | Moving laser | 119 | 610 - 1.2, 630 - 1.2, 740 - 1.4, 850 - 0.9
IN718  | Moving laser | 110 | 300 - 2.5, 400 - 3.0, 630 - 0.75, 630 - 1.25, 800 - 4.2
SS316  | Moving laser | 153 | 99 - 0.4, 300 - 0.4, 300 - 0.6, 408 - 0.4, 408 - 0.6
Ti64   | Moving laser | 153 | 111 - 0.4, 139 - 1.2, 154 - 0.4, 197 - 0.6, 311 - 0.7, 311 - 1.0,  520 - 1.0, 540 - 0.7
Ti64S  | Spot welding | 224 | 220, 270, 425

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
