"""
Network to load pretrained model from Magicleap.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchvision import models
import cv2
import numpy as np
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class SuperPointNet_resnet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet_resnet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # Shared Encoder.
    # c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    c3 = 64
    c4 = 128    # 512
    c5 = 256
    d1 = 256

    # Shared Encoder.
    resnet = models.resnet34(pretrained=True)
    self.firstconv = resnet.conv1
    self.firstbn = resnet.bn1
    self.firstrelu = resnet.relu
    self.firstmaxpool = resnet.maxpool
    self.encoder1 = resnet.layer1
    self.encoder2 = resnet.layer2

    self.conv1b = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # self.secondconv = resnet.conv1
    # self.secondbn = resnet.bn1
    # self.secondrelu =resnet.relu
    # self.secondmaxpool = resnet.maxpool

    # resnet.layer2[0].conv1 = nn.Conv2d(64, 128, 3, 1)
    # resnet.layer3[0].conv1 = nn.Conv2d(128, 256, 3, 1)
    # resnet.layer4[0].conv1 = nn.Conv2d(256, 512, 3, 1)

    # resnet.layer2[0].downsample[0] = nn.Conv2d(64, 128, 3, 1)
    # resnet.layer3[0].downsample[0] = nn.Conv2d(128, 256, 3, 1)
    # resnet.layer4[0].downsample[0] = nn.Conv2d(256, 512, 3, 1)

    self.encoder3 = resnet.layer3
    self.encoder4 = resnet.layer4
    #
    self.dblock = DACblock(512)
    self.spp = SPPblock(512)

    filters = [64, 128, 256, 512]

    self.decoder4 = DecoderBlock(516, filters[2])
    self.decoder3 = DecoderBlock(filters[2], filters[1])
    self.decoder2 = DecoderBlock(filters[1], filters[0])
    self.decoder1 = DecoderBlock(filters[0], filters[0])

    self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
    self.finalrelu1 = nonlinearity
    self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.finalrelu2 = nonlinearity
    self.finalconv3 = nn.Conv2d(32, 2, 3, padding=1)
    #
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    # plot test image

    # import matplotlib.pyplot as plt
    # test_img = x[0,:,:,:].permute(1, 2, 0)
    # test_warped_img = test_img.detach().cpu().numpy().copy()
    # # test_img_array = (test_warped_img * 255.0).astype(int)
    # plt.imshow(test_warped_img)
    # plt.show()
    # cv2.imwrite('test3.jpg', test_img_array)

    x = self.firstconv(x)
    x = self.firstbn(x)
    x = self.firstrelu(x)
    x = self.firstmaxpool(x)

    # x = self.secondconv(x)
    # x = self.secondbn(x)
    # x = self.secondrelu(x)
    # x = self.secondmaxpool(x)

    e1 = self.encoder1(x)
    e2 = self.encoder2(e1)
    # x = self.relu(self.conv1b(x))
    # x = self.relu(self.conv2a(x))
    # x = self.relu(self.conv2b(x))
    # x = self.pool(x)
    e3 = self.encoder3(e2)
    e4 = self.encoder4(e3)
    #
    e4 = self.dblock(e4)
    e4 = self.spp(e4)
    #
    d4 = self.decoder4(e4) + e3
    d3 = self.decoder3(d4) + e2
    d2 = self.decoder2(d3) + e1
    d1 = self.decoder1(d2)

    vessel = self.finaldeconv1(d1)
    vessel = self.finalrelu1(vessel)
    vessel = self.finalconv2(vessel)
    vessel = self.finalrelu2(vessel)
    vessel = self.finalconv3(vessel)
    vessel = F.sigmoid(vessel)

    # Detector Head.
    # cPa = self.relu(self.convPa(e4))
    cPa = self.relu(self.convPa(e2))
    semi = self.convPb(cPa)

    # Descriptor Head.
    # cDa = self.relu(self.convDa(e4))
    cDa = self.relu(self.convDa(e2))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

    output = {'semi': semi, 'desc': desc, 'vessel_feature': vessel}

    return output


###############################
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


###############################
class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


###############################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x