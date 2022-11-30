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

## The performance of semantic segmentation models
Metrics:
1. Loss function for all UNet and its variants is nn.CrossEntropyLoss
2. Loss function for all DeepLab and its varients is 70% nn.CrossEntropyLoss and 30% DiceBCELoss
3. The accuracies of all models are quantified by the intersetion over union (IoU) metric

| Model | train loss | val loss | test loss | #parameters | Test IoU |
| --- | --- | --- | --- | --- | --- |
| UNet (S) | 0.0016 | 0.0667 | 0.0360 | 17.2 M | **0.905** |
| UNet + Resnet50 (S) | 0.1305 | 0.0941 | 0.0620 | 32.5 M |  |
| UNet + Resnet50 (P) | 0.1545 | 0.0955 | 0.0561 | 32.5 M |  |
| UNet + MobileNetV2 (S) | 0.0652 | 0.0357 | 0.0383 | 6.6 M |  |
| UNet + MobileNetV2 (P) | 0.0865 | 0.0683 | 0.0565 | 6.6 M |  |
| UNet + Efficientnet-b5 (S) | 0.0017 | 0.139 | 0.0507 | 30.1 M |  |
| UNet + Efficientnet-b5 (P)| 0.0778 | 0.0372 | 0.0146 | 30.1 M | 0.897 |
| --- | --- | --- | --- | --- | --- |
| DeepLabV3 + ResNet50 (S) | 0.0258 | 0.0269 | 0.0270 | 39.6 M | 0.8733 |
| DeepLabV3 + ResNet50 (P) | 0.0330 | 0.0276 | 0.0272 | 39.6 M | 0.8754 |
| DeepLabV3 + MobileNetV2 (S) | 0.0273 | 0.0327 | 0.0299 | 12.6 M | 0.857 |
| DeepLabV3 + MobileNetV2 (P) | 0.0238 | 0.0326 | 0.0298 | 12.6 M | 0.863 |
| DeepLabV3 + Efficientnet-b5 (S) | | | | | |
| DeepLabV3 + Efficientnet-b5 (P) | | | | | |
|  |  |  |  |  |  |