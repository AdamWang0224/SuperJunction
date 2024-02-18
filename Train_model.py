"""This is the main training interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import random
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
import logging
from pathlib import Path

import loss
from utils.utils import dict_update
from utils.utils import flattenDetection
from utils.utils import extract_pnts
from utils.utils import labels2Dto3D
from utils.utils import save_checkpoint


class Train_model():
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # config
        # Update config
        print("Load Train_model!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.n_iter = 0
        self.max_iter = config["train_iter"]
        self.epoch = 0
        self.max_epoch = config["max_epoch"]
        self.n_epoch = config["n_epoch"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()

        self.train_loader = None
        self.val_loader = None
        pass

    def set_train_loader(self, loader):
        self.train_loader = loader

    def set_val_loader(self, loader):
        self.val_loader = loader

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set val loader")
        self._val_loader = loader

    def load_checkpoint(self, load_path, filename='checkpoint.pth.tar'):
        file_prefix = ['superPointNet']
        filename = '{}__{}'.format(file_prefix[0], filename)
        checkpoint = torch.load(load_path / filename)
        print("load checkpoint from ", filename)
        return checkpoint

    # mode: 'full' means the formats include the optimizer and epoch
    # full_path: if not full path, we need to go through another helper function
    def pretrainedLoader(self, net, optimizer, path, mode='full', full_path=False):
        # load checkpoint
        if full_path == True:
            checkpoint = torch.load(path)
        else:
            checkpoint = self.load_checkpoint(path)
        # apply checkpoint
        if mode == 'full':
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        return net, optimizer

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("adam optimizer")
        import torch.optim as optim

        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        # net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> creating model: %s", model)
        mod = __import__('models.{}'.format(model), fromlist=[''])
        net = getattr(mod, model)
        net = net(**params).to(self.device)

        logging.info("=> setting adam solver")
        optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])

        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
            pass
        else:
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            net, optimizer = self.pretrainedLoader(net, optimizer, path, mode=mode, full_path=True)
            logging.info("successfully load pretrained model from: %s", path)
            epoch = int(path.split('_')[-2])
            self.epoch = epoch + 1

        self.net = net
        self.optimizer = optimizer
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(self.net, lr=self.config["model"]["learning_rate"])
        pass

    def train(self):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max epoch
        :param options:
        :return:
        """
        # training info
        # logging.info("Current epoch: %d", self.epoch)
        logging.info("Max epoch: %d", self.max_epoch)

        while self.epoch <= self.max_epoch:
            if self.epoch % self.n_epoch == 0:
                f = open('./training_output.txt', 'a+')
                f.write('epoch: ' + str(self.epoch) + '\n')
                f.close()
            logging.info("Current epoch: %d", self.epoch)

            running_losses = []
            running_losses_detector = []
            running_loss_descriptor = []
            running_loss_vessel = []

            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # extract keypoints and warped keypints
                pnts, warped_pnts = extract_pnts(sample_train["labels_2D"], sample_train['homographies'])
                loss_out, detector_loss, descriptor_loss, vessel_loss = self.train_val_sample(sample_train, pnts,
                                                                                              warped_pnts, self.epoch,
                                                                                              True)
                running_losses.append(loss_out)
                running_losses_detector.append(detector_loss)
                running_loss_descriptor.append(descriptor_loss)
                running_loss_vessel.append(vessel_loss)

            if self.epoch % self.n_epoch == 0:
                avg_running_loss = np.average(running_losses)
                avg_running_loss_detector = np.average(running_losses_detector)
                avg_running_loss_descriptor = np.average(running_loss_descriptor)
                avg_running_loss_vessel = np.average(running_loss_vessel)
                f = open('training_output.txt', 'a+')
                f.write('train loss: ' + str(avg_running_loss) + '\n')
                f.write('train loss detector: ' + str(avg_running_loss_detector) + '\n')
                f.write('train loss descriptor: ' + str(avg_running_loss_descriptor) + '\n')
                f.write('train loss vessel: ' + str(avg_running_loss_vessel) + '\n')
                f.close()

                # run validation
                logging.info("====== Validating...")
                running_losses = []
                running_losses_detector = []
                running_loss_descriptor = []
                running_loss_vessel = []
                for j, sample_val in enumerate(self.val_loader):
                    # extract keypoints and warped keypints
                    pnts, warped_pnts = extract_pnts(sample_val["labels_2D"], sample_val['homographies'])
                    loss_out, detector_loss, descriptor_loss, vessel_loss = self.train_val_sample(sample_val, pnts,
                                                                                                  warped_pnts,
                                                                                                  self.epoch,
                                                                                                  False)
                    running_losses.append(loss_out)
                    running_losses_detector.append(detector_loss)
                    running_loss_descriptor.append(descriptor_loss)
                    running_loss_vessel.append(vessel_loss)
                avg_running_loss = np.average(running_losses)
                avg_running_loss_detector = np.average(running_losses_detector)
                avg_running_loss_descriptor = np.average(running_loss_descriptor)
                avg_running_loss_vessel = np.average(running_loss_vessel)
                f = open('training_output.txt', 'a+')
                f.write('val loss: ' + str(avg_running_loss) + '\n')
                f.write('val loss detector: ' + str(avg_running_loss_detector) + '\n')
                f.write('val loss descriptor: ' + str(avg_running_loss_descriptor) + '\n')
                f.write('val loss vessel: ' + str(avg_running_loss_vessel) + '\n')
                f.write('\n')
                f.close()

                # save model
                logging.info(f"save model: every {self.n_epoch} epoch, current epoch: {self.epoch}")
                self.saveModel(self.epoch)

            # ending condition
            if self.epoch > self.max_epoch:
                # end training
                logging.info("End training: %d", self.epoch)
                break
            else:
                self.epoch += 1

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)

        return loss

    def vessel_loss(self, target, input):
        loss_func = loss.FocalLoss(gamma=2, alpha=0.25)
        vessel_loss = loss_func(target, input)

        return vessel_loss

    def sample_keypoint_desc(self, keypoints, descriptors, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape

        keypoints = keypoints.clone().float()

        keypoints /= torch.tensor([(w * s - 1), (h * s - 1)]).to(keypoints)[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2).to(self.device), mode='bilinear', **args)

        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def sample_descriptors(self, pnts, descriptor_pred, scale=8):
        """extract descriptors based on keypoints"""
        descriptors = [self.sample_keypoint_desc(k[None], d[None], s=scale)[0]
                       for k, d in zip(pnts, descriptor_pred)]

        return descriptors

    def pairwise_distance(self, x1, x2, p=2, eps=1e-6):
        r"""
        Computes the batchwise pairwise distance between vectors v1,v2:
            .. math ::
                \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}
            Args:
                x1: first input tensor
                x2: second input tensor
                p: the norm degree. Default: 2
            Shape:
                - Input: :math:`(N, D)` where `D = vector dimension`
                - Output: :math:`(N, 1)`
            >>> input1 = autograd.Variable(torch.randn(100, 128))
            >>> input2 = autograd.Variable(torch.randn(100, 128))
            >>> output = F.pairwise_distance(input1, input2, p=2)
            >>> output.backward()
        """
        assert x1.size() == x2.size(), "Input sizes must be equal."
        assert x1.dim() == 2, "Input must be a 2D matrix."

        return 1 - torch.cosine_similarity(x1, x2, dim=1)
        # diff = torch.abs(x1 - x2)
        # out = torch.sum(torch.pow(diff + eps, p), dim=1)
        #
        # return torch.pow(out, 1. / p)

    def triplet_margin_loss_gor(self, anchor, positive, negative1, negative2, beta=1.0, margin=1.0, p=2, eps=1e-6,
                                swap=False):
        assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
        assert anchor.size() == negative1.size(), "Input sizes between anchor and negative must be equal."
        assert positive.size() == negative2.size(), "Input sizes between positive and negative must be equal."
        assert anchor.dim() == 2, "Inputd must be a 2D matrix."
        assert margin > 0.0, 'Margin should be positive value.'

        # loss1 = triplet_margin_loss_gor_one(anchor, positive, negative1)
        # loss2 = triplet_margin_loss_gor_one(anchor, positive, negative2)
        #
        # return 0.5*(loss1+loss2)

        d_p = self.pairwise_distance(anchor, positive, p, eps)
        d_n1 = self.pairwise_distance(anchor, negative1, p, eps)
        d_n2 = self.pairwise_distance(anchor, negative2, p, eps)  # original

        dist_hinge = torch.clamp(margin + d_p - 0.5 * (d_n1 + d_n2), min=0.0) # original
        # dist_hinge = torch.clamp(margin + d_p / d_n1, min=0.0)  # ours
        # dist_hinge = torch.clamp(margin + d_p - d_n1, min=0.0)  # ours

        neg_dis1 = torch.pow(torch.sum(torch.mul(anchor, negative1), 1), 2)
        gor1 = torch.mean(neg_dis1)
        neg_dis2 = torch.pow(torch.sum(torch.mul(anchor, negative2), 1), 2)   # original
        gor2 = torch.mean(neg_dis2)   # original

        loss = torch.mean(dist_hinge) + beta * (gor1 + gor2)   # original
        # loss = torch.mean(dist_hinge) + beta * gor1 # ours
        return loss

    def descriptor_loss_triplet(self, pnts, warped_pnts, descriptor_pred, warped_descriptor_pred, topk):
        """compute triplet descriptor loss"""
        descriptors = self.sample_descriptors(pnts, descriptor_pred)
        warped_descriptors = self.sample_descriptors(warped_pnts, warped_descriptor_pred)

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        for i in range(len(warped_descriptors)):
            if warped_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = warped_descriptors[i]

            n = warped_descriptors[i].shape[1]
            if n > 1000:  # avoid OOM
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)
            ar = torch.arange(n)

            # random
            neg_index2 = []
            if n == 1:
                neg_index2.append(0)
            else:
                for j in range(n):
                    t = j
                    while t == j:
                        t = random.randint(0, n - 1)
                    neg_index2.append(t)
            neg_index2 = torch.tensor(neg_index2, dtype=torch.long).to(affine_descriptor)

            # hard
            with torch.no_grad():
                dis = torch.norm(descriptor - affine_descriptor, dim=0)
                dis[ar, ar] = dis.max() + 1
                neg_index1_temp = dis.argmin(axis=1)

                # wangyu start: to replace neg_index1 with relaxed hard
                if topk > dis.shape[0]:
                    topk = dis.shape[0]
                # candidate = dis.sort(axis=1, descending=False)[1][:, :n]
                # # col_indices = torch.randint(0, n, size=(candidate.shape[0], 1)).to("cuda:0")
                # col_indices = torch.randint(0, n, size=(candidate.shape[0], 1)).to(candidate)
                # col_indices = col_indices.type(torch.int64)
                # neg_index1 = torch.gather(candidate, 1, col_indices).flatten()
                candidate = dis.sort(axis=1, descending=False)[1][:, :topk].to(affine_descriptor)
                col_indices = torch.randint(0, topk, size=(candidate.shape[0], 1)).to(affine_descriptor)
                col_indices = col_indices.type(torch.int64)
                neg_index1 = torch.gather(candidate, 1, col_indices).flatten()
                neg_index1 = torch.tensor(neg_index1, dtype=torch.long).to(affine_descriptor)
                # wangyu end

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))

        if len(positive) == 0:
            return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

        positive = torch.cat(positive)
        anchor = torch.cat(anchor)
        negatives_hard = torch.cat(negatives_hard)
        negatives_random = torch.cat(negatives_random)

        positive = F.normalize(positive, dim=-1, p=2)
        anchor = F.normalize(anchor, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_random = F.normalize(negatives_random, dim=-1, p=2)

        loss = self.triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)

        return loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(mask_2D.to(device), cell_size=cell_size, add_dustbin=False).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    def train_val_sample(self, sample, pnts, warped_pnts, epoch, train=False):
        """
        # key function
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        task = "train" if train else "val"
        if_warp = self.config['data']['warped_pair']['enable']

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}

        # original image
        img, labels_2D, mask_2D, vessel = sample["image"], sample["labels_2D"], sample["valid_mask"], sample["vessel"]

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

        # warped images
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = sample["warped_img"], sample["warped_labels"], sample["warped_valid_mask"]
            # homographies
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            outs = self.net(img.to(self.device))
            semi, coarse_desc, vessel_feature = outs["semi"], outs["desc"], outs["vessel_feature"]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
        else:
            with torch.no_grad():
                outs = self.net(img.to(self.device))
                semi, coarse_desc, vessel_feature = outs["semi"], outs["desc"], outs["vessel_feature"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                    # semi_warp, coarse_desc_warp, vessel_feature = outs_warp["semi"], outs_warp["desc"], outs["vessel_feature"]
                pass

        # detector loss
        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
            if if_warp:
                warped_labels = sample["warped_labels_gaussian"]
        else:
            labels_2D = sample["labels_2D"]
            if if_warp:
                warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        # detector loss for original image
        labels_3D = labels2Dto3D(labels_2D.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin).float()
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.detector_loss(input=outs["semi"], target=labels_3D.to(self.device),
                                      mask=mask_3D_flattened, loss_type=det_loss_type)

        # detector loss for warped image
        if if_warp:
            labels_3D = labels2Dto3D(warped_labels.to(self.device),
                                     cell_size=self.cell_size, add_dustbin=add_dustbin).float()
            mask_3D_flattened = self.getMasks(mask_warp_2D, self.cell_size, device=self.device)
            loss_det_warp = self.detector_loss(input=outs_warp["semi"], target=labels_3D.to(self.device),
                                               mask=mask_3D_flattened, loss_type=det_loss_type)
        else:
            loss_det_warp = torch.tensor([0]).float().to(self.device)

        loss = loss_det + loss_det_warp

        # descriptor loss
        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]
        topk = self.config["model"]["topk"]
        if lambda_loss > 0:
            assert if_warp is True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(coarse_desc, coarse_desc_warp,
                                                                                 mat_H, mask_valid=mask_desc,
                                                                                 device=self.device, **self.desc_params)
            triplet_loss_desc = self.descriptor_loss_triplet(pnts, warped_pnts,
                                                             coarse_desc, coarse_desc_warp, topk)
            loss += lambda_loss * triplet_loss_desc
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze
            triplet_loss_desc = ze

        # vessel loss
        vessel = vessel[:, :1, :, :]
        loss_vessel = self.vessel_loss(vessel.to(self.device), vessel_feature)
        lambda_vessel_loss = self.config["model"]["lambda_vessel_loss"]
        loss += lambda_vessel_loss * loss_vessel

        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "loss_vessel": loss_vessel,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
                "triplet_loss_desc": triplet_loss_desc
            }
        )

        if train:
            loss.backward()
            self.optimizer.step()

        if epoch % self.n_epoch == 0 or task == "val":
            print(f"{task} loss: {self.scalar_dict['loss'].item()}")

        detector_loss = self.scalar_dict['loss_det'].item() + self.scalar_dict['loss_det_warp'].item()
        # descriptor_loss = self.scalar_dict['positive_dist'].item() + self.scalar_dict['negative_dist'].item()
        descriptor_loss = self.scalar_dict['triplet_loss_desc'].item()
        vessel_loss = self.scalar_dict['loss_vessel'].item()

        return loss.item(), detector_loss, descriptor_loss, vessel_loss

    ######## static methods ########
    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input: 
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    def saveModel(self, epoch):
        """
        # save checkpoint for resuming training
        :return:
        """
        model_state_dict = self.net.module.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            epoch
            # self.n_iter,
        )
        pass

