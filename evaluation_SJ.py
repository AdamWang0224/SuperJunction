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

def evaluate(args, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = args.path
    # files = find_files_with_ext(path)
    files = os.listdir(path)
    top_K = 1000
    print("top_K: ", top_K)

    # create output dir
    path_img_pts = path + '/img_with_pts'
    os.makedirs(path_img_pts, exist_ok=True)

    # for i in range(2):
    #     f = files[i]
    print(f"file: {files[0]}")
    # files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils import saveImg

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        # img_num = int(f_num) - 1
        # img = cv2.imread('./datasets/retina_test/img/'+str(img_num)+'.jpg') # 2912x2912, 0-255
        img = cv2.imread('./datasets/retina_test/img/' + f_num + '.jpg')
        # img = cv2.imread('./datasets/retina_test/img/' + f_num + '.bmp')
        # resize image
        dim = (1024, 1024)
        # dim = (512, 384)

        # resize image
        resized = cv2.resize(img, dim)
        print(resized.max())
        # cv2.imwrite('test.jpg', resized)

        # real_H = data['homography']
        image = data['image']   # 1024x1024, 0-1
        # warped_image = data['warped_image']
        # keypoints = data['prob'][:, [1, 0]]
        # print("keypoints: ", keypoints[:3,:])
        # warped_keypoints = data['warped_prob'][:, [1, 0]]
        # print("warped_keypoints: ", warped_keypoints[:3,:])

        pts = data['prob'][:, :2]
        pts = pts.tolist()

        # overlay key points on image_thresh
        for pt in pts:
            cv2.circle(resized, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        print(resized)
        path_to_output_img = path + '/img_with_pts/' + f_num + '.png'
        cv2.imwrite(path_to_output_img, resized)



if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sift', action='store_true', help='use sift matches')
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-i', '--img_with_pts', action='store_true')
    parser.add_argument('-r', '--repeatibility', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    args = parser.parse_args()
    evaluate(args)
