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
from pathlib import Path
from models.net_util import AGGGate

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

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_w = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (width - border))
    mask_h = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (height - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def remove_borders_projected(keypoints, scores, border, height, width, homo_matrix):
    projected_keypoints = warp_keypoints(keypoints, torch.inverse(homo_matrix))
    mask_w = (projected_keypoints[:, 0] >= border) & (projected_keypoints[:, 0] < (width - border))
    mask_h = (projected_keypoints[:, 1] >= border) & (projected_keypoints[:, 1] < (height - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def warp_keypoints(keypoints, homography_mat):
    source = torch.cat([keypoints, torch.ones(len(keypoints), 1).to(keypoints.device)], dim=-1)
    dest = (homography_mat @ source.T).T
    # dest /= dest[:, 2:3]
    dest_clone = torch.clone(dest)
    dest = dest_clone / dest_clone[:, 2:3]
    return dest[:, :2]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPointNet_resnet_gate(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0, #0.005
        'max_keypoints': -1,
        'remove_borders': 4
    }
    def __init__(self, config):
        super(SuperPointNet_resnet_gate, self).__init__()
        self.config = {**self.default_config, **config}

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

        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        filters = [64, 128, 256, 512]

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 8, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(16, 2, 3, padding=1)
        #
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.aggregation = AGGGate(rgb_in_planes=8, disp_in_planes=1, bn_momentum=0.1)

        path = Path(__file__).parent / 'weights/superPointNet_index_4_81_checkpoint.pth.tar'
        # self.load_state_dict(torch.load(str(path)))
        checkpoint = torch.load(str(path))
        self.load_state_dict(checkpoint['model_state_dict'])

        print('Loaded SuperPoint model')

    def forward(self, data, curr_max_kp=None, curr_key_thresh=None):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if curr_max_kp is None:
            curr_max_kp = self.config['max_keypoints']
        curr_max_kp = 512   # 512
        if curr_key_thresh is None:
            curr_key_thresh = self.config['keypoint_threshold']

        # Shared Encoder.
        x = self.firstconv(data['image'])
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        cPa = self.relu(self.convPa(e2))
        scores = self.convPb(cPa)

        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if curr_max_kp >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, curr_max_kp)
                for k, s in zip(keypoints, scores)]))

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(e2))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # Extract descriptors
        desc = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, desc)]

        return {'keypoints': keypoints, 'scores': scores, 'descriptors': desc}

    def forward_train(self, data):
        homo_matrices = data['homography']

        x = self.firstconv(data['image'])
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        cPa = self.relu(self.convPa(e2))
        scores = self.convPb(cPa)

        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Discard keypoints near the image borders
        homo_mat_index = 0
        results = []
        mid_point = len(keypoints) // 2
        for i, (k,s) in enumerate(zip(keypoints, scores)):
            if i < mid_point: # orig image
                results.append(remove_borders(k, s, self.config['remove_borders'], h*8, w*8))
            else:
                homo_matrix = homo_matrices[homo_mat_index]
                homo_mat_index += 1
                results.append(remove_borders_projected(k, s, self.config['remove_borders'], h*8, w*8, homo_matrix))
        keypoints, scores = list(zip(*results))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        keypoints, scores = list(keypoints), list(scores)

        for i, (k, s) in enumerate(zip(keypoints, scores)):
            """
            Occurence of below condition is very rare as we are sampling keypoints above threshold of 0 itself and then
            sampling max_keypoints from it. But incase if it happends then we are randomly adding some pixel locations to 
            without checking any conditions with respect to preexisting keypoints.
            """
            # print('Number of keypoints: {}'.format(len(k)))
            if len(k) < self.config['max_keypoints']:
                print("Rare condition executed")
                to_add_points = self.config['max_keypoints'] - len(k)
                random_keypoints = torch.stack(
                    [torch.randint(0, w * 8, (to_add_points,), dtype=torch.float32, device=k.device),
                     torch.randint(0, h * 8, (to_add_points,), dtype=torch.float32, device=k.device)], 1)
                keypoints[i] = torch.cat([keypoints[i], random_keypoints], dim=0)
                scores[i] = torch.cat(
                    [scores[i], torch.ones(to_add_points, dtype=torch.float32, device=s.device) * 0.1], dim=0)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(e2))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # Extract descriptors
        desc = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, desc)]

        return {'keypoints': keypoints, 'scores': scores, 'descriptors': desc}

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