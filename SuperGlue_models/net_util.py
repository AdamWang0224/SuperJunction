import sys
from collections import OrderedDict
import torch.nn as nn
import torch
import functools

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(in_planes, out_planes // reduction),
            # nn.ReLU(inplace=True),
            # nn.Linear(out_planes // reduction, out_planes),
            nn.Linear(in_planes, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

'''
Feature Separation Part
'''
class FSP(nn.Module):
    def __init__(self, guide_in_planes, main_in_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(in_planes=guide_in_planes+main_in_planes, out_planes=main_in_planes)
        # self.filter = FilterLayer(5, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = channel_weight * mainPath
        return out

'''
SA-Gate
'''
class SAGate(nn.Module):
    def __init__(self, rgb_in_planes, disp_in_planes, bn_momentum=0.0003):
        self.init__ = super(SAGate, self).__init__()
        reduction = 16
        self.rgb_in_planes = rgb_in_planes
        self.disp_in_planes = disp_in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(self.disp_in_planes, self.rgb_in_planes, reduction)
        self.fsp_disp = FSP(self.rgb_in_planes, self.disp_in_planes, reduction)

        self.gate_rgb = nn.Conv2d(self.rgb_in_planes+self.disp_in_planes, 1, kernel_size=1, bias=True)
        self.gate_disp = nn.Conv2d(self.rgb_in_planes+self.disp_in_planes, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb, disp = x

        rec_rgb = self.fsp_rgb(disp, rgb)
        rec_disp = self.fsp_disp(rgb, disp)
        cat_fea = torch.cat([rec_rgb, rec_disp], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_disp(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb*attention_vector_l + disp*attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        disp_out = (disp + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        disp_out = self.relu2(disp_out)

        concat_fea = torch.cat([rgb_out, disp_out], dim=1)

        return [rgb_out, disp_out], concat_fea

class AGGGate(nn.Module):
    def __init__(self, rgb_in_planes, disp_in_planes, bn_momentum=0.0003):
        self.init__ = super(AGGGate, self).__init__()
        reduction = 16
        self.rgb_in_planes = rgb_in_planes
        self.disp_in_planes = disp_in_planes
        self.bn_momentum = bn_momentum

        # self.fsp_rgb = FSP(self.disp_in_planes, self.rgb_in_planes, reduction)
        # self.fsp_disp = FSP(self.rgb_in_planes, self.disp_in_planes, reduction)

        self.gate_rgb = nn.Conv2d(self.rgb_in_planes+self.disp_in_planes, 1, kernel_size=1, bias=True)
        self.gate_disp = nn.Conv2d(self.rgb_in_planes+self.disp_in_planes, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb, disp = x

        # rec_rgb = self.fsp_rgb(disp, rgb)
        # rec_disp = self.fsp_disp(rgb, disp)
        cat_fea = torch.cat([rgb, disp], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_disp(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb*attention_vector_l + disp*attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        disp_out = (disp + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        disp_out = self.relu2(disp_out)

        concat_fea = torch.cat([rgb_out, disp_out], dim=1)

        return [rgb_out, disp_out], concat_fea