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
The original data come from 32ID APS at Argonne National Laboratory。

Material | Type | Power (W) | Scan speed (mm/s) | Number of images
--- | --- | --- | --- | ---
Al6061 | Moving laser | x | x | x
IN718 | x | x | x | x
SS304 | x | x | x | x
SS316 | x | x | x | x
Ti64 | x | x | x |x 
Ti64(Spot welding) | x | x | x |x 

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
