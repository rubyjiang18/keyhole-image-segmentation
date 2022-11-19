# keyhole-image-segmentation
This is the repo for keyhole image segmentation

## Keyhole Segmentation Models
1. Unet
2. Unet + ResNet50
3. Unet + MobileNet
5. Deeplab v3
6. Deeplab + ResNet 
7. DeepLab + MobileNet

## Papers
1. Unet: https://github.com/milesial/Pytorch-UNet#weights--biases
2. Unet + ResNet: https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py; https://github.com/usuyama/pytorch-unet (unet+resnet18)
3. Unet + MobileNetV3: https://github.com/akinoriosamura/mobile-segmentation-mobilenet-unet Condition: input image size 224 * 224? why
(recepie: epoch=150, lr=0.05, cosine learning rate, weight decay=4e-5, remove dropout: https://github.com/tonylins/pytorch-mobilenet-v2)
4. DeeplabV3 + ResNet50 : https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
5. DeepLabV3 + MobileNetV3: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/

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
