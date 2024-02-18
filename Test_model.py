import torch
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import cv2

from utils.utils import flattenDetection, squeezeToNumpy


@torch.no_grad()
class Test_model():
    def __init__(self, config, device='cpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device = device

        ## other parameters
        self.nms_dist = self.config['nms']
        self.conf_thresh = self.config['detection_threshold']
        self.nn_thresh = self.config['nn_thresh']  # L2 descriptor distance for good match.
        self.cell = 8  # deprecated
        self.cell_size = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None # np[batch, 1, H, W]
        self.pts = None
        self.pts_subpixel = None
        ## new variables
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None

        self.test_loader = None

        self.subpixel = self.config["subpixel"]["enable"]
        self.patch_size = self.config["subpixel"]["patch_size"]
        pass

    def set_test_loader(self, loader):
        self.test_loader = loader

    @property
    def test_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._test_loader

    @test_loader.setter
    def test_loader(self, loader):
        print("set test loader")
        self._test_loader = loader


    def loadModel(self):
        logging.info("=> creating model: %s", self.model)
        mod = __import__('models.{}'.format(self.model), fromlist=[''])
        net = getattr(mod, self.model)
        self.net = net(**self.params)

        checkpoint = torch.load(self.weights_path,
                                map_location=lambda storage, loc: storage)
        if '.tar' in self.weights_path:
            self.net.load_state_dict(checkpoint['model_state_dict'])  # for SuperPointNet_renet
        else:
            self.net.load_state_dict(checkpoint)    # for SuperPointNet

        self.net = self.net.to(self.device)
        logging.info('successfully load pretrained model from: %s', self.weights_path)
        pass

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds


    def getPtsFromHeatmap(self, heatmap):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        heatmap = heatmap.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        self.sparsemap = (heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts


    def heatmap_to_pts(self):
        heatmap_np = self.heatmap

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np] # [batch, H, W]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch

    def run(self, images):
        """
        input:
            images: tensor[batch(1), 1, H, W]

        """
        from Train_model import Train_model
        from utils.utils import toNumpy
        train_agent = Train_model

        with torch.no_grad():
            outs = self.net(images)
        semi = outs['semi']
        self.outs = outs

        channel = semi.shape[1]
        if channel == 64:
            heatmap = train_agent.flatten_64to1(semi, cell_size=self.cell_size)
        elif channel == 65:
            heatmap = flattenDetection(semi, tensor=True)

        heatmap_np = toNumpy(heatmap)
        self.heatmap = heatmap_np
        return self.heatmap

    def soft_argmax_points(self, pts, patch_size=5):
        """
        input:
            pts: tensor [N x 2]
        """
        from utils.losses import extract_patch_from_points
        from utils.losses import soft_argmax_2d
        from utils.losses import norm_patches

        ##### check not take care of batch #####
        # print("not take care of batch! only take first element!")
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(self.heatmap, pts, patch_size=patch_size)
        import torch
        patches = np.stack(patches)
        patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)

        # norm patches
        patches_torch = norm_patches(patches_torch)

        from utils.losses import do_log
        patches_torch = do_log(patches_torch)

        dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
        # print("dxdy: ", dxdy.shape)
        points = pts
        points[:,:2] = points[:,:2] + dxdy.numpy().squeeze() - patch_size//2
        self.patches = patches_torch.numpy().squeeze()
        self.pts_subpixel = [points.transpose().copy()]
        return self.pts_subpixel.copy()

    def sample_desc_from_points(self, coarse_desc, pts):
        # --- Process descriptor.
        H, W = coarse_desc.shape[2]*self.cell, coarse_desc.shape[3]*self.cell
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc

    def desc_to_sparseDesc(self):
        # pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        desc_sparse_batch = [self.sample_desc_from_points(self.outs['desc'], pts) for pts in self.pts_nms_batch]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch

    def test(self, output_dir):
        output_dir_pts = output_dir + 'predictions'
        os.makedirs(output_dir_pts, exist_ok=True)

        output_dir_img = output_dir + 'img_with_pts'
        os.makedirs(output_dir_img, exist_ok=True)

        precision_lst = []
        recall_lst = []
        for i, sample in tqdm(enumerate(self.test_loader)):
            img_0 = sample["image"]
            image_name = sample['image_name']

            heatmap_batch = self.run(img_0.to(self.device))

            # heatmap to pts
            pts = self.heatmap_to_pts()

            if self.subpixel:
                pts = self.soft_argmax_points(pts, patch_size=self.patch_size)

            # heatmap, pts to desc
            desc_sparse = self.desc_to_sparseDesc()

            pts = pts[0]
            desc = desc_sparse[0]

            pts = pts.transpose()
            desc = desc.transpose()
            threshold = 0.015  # filter away key points with lower probability
            desc = desc[pts[:, 2] >= threshold]
            pts = pts[pts[:, 2] >= threshold]

            # save keypoints
            pred = {"image": squeezeToNumpy(img_0)}
            image = np.transpose(squeezeToNumpy(img_0), (1, 2, 0))
            pred.update({"prob": pts, "desc": desc})
            # pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

            filename = image_name[0].split('.')[0]
            path = Path(output_dir_pts, "{}.npz".format(filename))
            # print(path)
            np.savez_compressed(path, **pred)

            # draw image with keypints
            path = Path(output_dir_img, "{}.png".format(filename))
            self.draw(filename, pts, path)

            # # calculate precision and recall
            # precision, recall = self.calc_precision_recall(filename, pts)
            # precision_lst.append(precision)
            # recall_lst.append(recall)

        print('Average precision: ', np.average(precision_lst))
        print('Average recall: ', np.average(recall_lst))

    def draw(self, image_name, pts, path):
        img = cv2.imread('./datasets/retina_test/img/' + image_name + '.jpg')
        dim = (1024, 1024)
        resized = cv2.resize(img, dim)
        print(resized.max())
        print(resized.min())
        print(resized.mean())
        print(resized.shape)

        # overlay key points on image_thresh
        for pt in pts:
            cv2.circle(resized, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        cv2.imwrite(str(path), resized)
        print()

    def calc_precision_recall(self, filename, pts):
        path_to_ground_truth = './datasets/retina_test/pts/'
        path_to_ground_truth_file = path_to_ground_truth + filename + '.npz'
        data_ground_truth = np.load(path_to_ground_truth_file)
        pts_ground_truth = data_ground_truth['pts']

        threshold = 0.015
        pts_prediction = pts[pts[:, 2] >= threshold]

        threshold_matching = 5
        tp_lst = []
        for i in range(pts_ground_truth.shape[0]):
            pt_gt = pts_ground_truth[i]
            pt_gt = list(pt_gt)
            for j in range(pts_prediction.shape[0]):
                pt_pred = pts_prediction[j]
                pt_pred = list(pt_pred)
                dist = np.sqrt((pt_gt[0] - pt_pred[0]) ** 2 + (pt_gt[1] - pt_pred[1]) ** 2)
                if (dist <= threshold_matching) and (pt_pred not in tp_lst):
                    tp_lst.append(pt_pred)
                    break
        true_positive = len(tp_lst)
        false_positive = pts_prediction.shape[0] - len(tp_lst)
        false_negative = pts_ground_truth.shape[0] - len(tp_lst)

        if pts_prediction.shape[0] == 0:
            precision = 0
            recall = 0
        else:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

        return precision, recall

