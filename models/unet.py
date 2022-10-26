""" 
Full assembly of the parts to form the complete network 
"""

from distutils.command.build_clib import build_clib
from unet_parts import *

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.biliear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = outConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x2)
        x3 = self.down2(x1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logit = self.outc(x)
        return logit



###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 572, 572)
    model = Unet()

    print(list(model.children()))
    import time
    t = time.time()
    x = model(im)
    print(time.time() - t)
    print(x.shape)
    del model
    del x