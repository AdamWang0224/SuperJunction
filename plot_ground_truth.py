"""Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability
import cv2
import matplotlib.pyplot as plt

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs
from utils.logging import *

def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    inliers = data['inliers'].astype(bool)
    # matches = np.array(data['matches'])[inliers].tolist()
    # matches = matches[inliers].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

if __name__ == "__main__":
    # path_to_img = './logs/superpoint_retina_240x320_large_dataset_0.00001/predictions_checkpoint_200000_0%/'
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = './logs/superpoint_retina_test/predictions_100_0%'
    files = find_files_with_ext(path)
    top_K = 1000
    print("top_K: ", top_K)

    files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils_color import saveImg

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        image = data['image']

        data_ground_truth = np.load('./datasets/retina_test/pts/' + f)
        keypoints = data_ground_truth['pts'][:, [1, 0]]
        print("keypoints: ", keypoints[:3,:])
        # warped_keypoints = data['warped_prob'][:, [1, 0]]
        # print("warped_keypoints: ", warped_keypoints[:3,:])

        pts = data_ground_truth['pts']
        img = draw_keypoints(image*255, pts.transpose())

        plot_imgs([img.astype(np.uint8)], titles=['ground_truth'], dpi=200)
        # plt.title("rep: " + str(repeatability[-1]))
        plt.tight_layout()

        plt.savefig('./logs/superpoint_retina_test/img_with_ground_truth_pts/' + f_num + '.png', dpi=300, bbox_inches='tight')
        pass

