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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SuperPointNet_resnet_enhance(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self, n_class=1):
    super(SuperPointNet_resnet_enhance, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # Shared Encoder.
    # c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    c1 = 64
    c2 = 64
    c3 = 64
    c4 = 128    # 512
    c5 = 256
    d1 = 256
    d2 = 256

    # Shared Encoder.
    resnet = models.resnet34(pretrained=True)
    self.firstconv = resnet.conv1
    self.firstbn = resnet.bn1
    self.firstrelu = resnet.relu
    self.firstmaxpool = resnet.maxpool
    self.encoder1 = resnet.layer1
    self.encoder2 = resnet.layer2

    self.conv1a = torch.nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)

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

    self.ddecoder3 = DecoderBlock(filters[3], filters[1])
    self.ddecoder2 = DecoderBlock(filters[2], filters[0])
    self.ddecoder1 = DecoderBlock(filters[1], filters[0])
    self.ddecoder0 = DecoderBlock(filters[1], filters[0])

 #   self.dsdecoder3 = DecoderBlock(filters[3], filters[2])
   # self.dsdecoder2 = DecoderBlock(filters[2]+filters[1], filters[2])
 #   self.dsdecoder1 = DecoderBlock(filters[2]+filters[0], filters[2])
    # self.dsdecoder0 = DecoderBlock(filters[2]+filters[0], filters[2])

    self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
    self.finalrelu1 = nonlinearity
    self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.finalrelu2 = nonlinearity
    self.finalconv3 = nn.Conv2d(32, 2, 3, padding=1)
    #
    # Detector Head.
    #
    # self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    # self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # self.dconv_up3 = double_conv(c3 + c4, c3)
    # self.dconv_up3 = double_conv(516+c5, c5)
    # # self.dconv_up2 = double_conv(c2 + c3, c2)
    # self.dconv_up2 = double_conv(c5+c4, c4)
    # # self.dconv_up1 = double_conv(c1 + c2, c1)
    # self.dconv_up1 = double_conv(c4+c3, c3)
    # self.dconv_up0 = double_conv(c3+c2, c2)
    self.trans_conv = nn.ConvTranspose2d(256, 256, 2, stride=16)
    self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

    # Descriptor Head.
    # self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    # self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
    # self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)
    # self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

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
   # conv1 = self.relu(self.conv1a(x))

    x = self.firstconv(x)
    x = self.firstbn(x)
    conv1 = self.firstrelu(x)
    x = self.firstmaxpool(conv1)

    e1 = self.encoder1(x)
    e2 = self.encoder2(e1)
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

    conv2 = d1
    conv3 = self.relu(self.conv1b(e1))

    vessel = self.finaldeconv1(d1)
    vessel = self.finalrelu1(vessel)
    vessel = self.finalconv2(vessel)
    vessel = self.finalrelu2(vessel)
    vessel = self.finalconv3(vessel)
    vessel = F.sigmoid(vessel)

    # Detector Head.
    # cPa = self.relu(self.convPa(e2))
    # semi = self.convPb(cPa)
    cPa = self.decoder4(e4)
    cPa = torch.cat([cPa, e3], dim=1)

    cPa = self.ddecoder3(cPa)
    cPa = torch.cat([cPa, e2], dim=1)

    cPa = self.ddecoder2(cPa)
    cPa = torch.cat([cPa, e1], dim=1)

    cPa = self.ddecoder1(cPa)
    cPa = torch.cat([cPa, conv1], dim=1)

    cPa = self.ddecoder0(cPa)

    semi = self.conv_last(cPa)
    semi = torch.sigmoid(semi)

    # Descriptor Head.
    cDa = self.decoder4(e4)
    cDa = torch.cat([cDa, e3], dim=1)

    # cDa = self.dsdecoder3(cDa)
    # cDa = torch.cat([cDa, e2], dim=1)

    # cDa = self.dsdecoder2(cDa)
    # cDa = torch.cat([cDa, e1], dim=1)

    # cDa = self.dsdecoder1(cDa)
    # cDa = torch.cat([cDa, conv1], dim=1)
    cDa = self.upsample(cDa)
    cDa = self.upsample(cDa)
    cDa = self.upsample(cDa)
    desc = self.upsample(cDa)
    #desc = self.dsdecoder0(cDa)

    # cDa = self.upsample(e2)
    # cDa = self.relu(self.convDa(cDa))
    # cDa = self.upsample(cDa)
    # desc = self.relu(self.convDb(cDa))
    # # cDa = self.upsample(cDa)
    # # # desc = self.convDc(cDa)
    # # desc = cDa
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    # desc = self.trans_conv(desc)

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