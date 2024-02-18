"""Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
from pathlib import Path
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability
import cv2
import matplotlib.pyplot as plt

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs
from utils.logging import *

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

def extract(args, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = args.path
    files = find_files_with_ext(path)
    top_K = 1000
    print("top_K: ", top_K)

    # create output dir
    path_pts = path + '/pts'
    os.makedirs(path_pts, exist_ok=True)

    # for i in range(2):
    #     f = files[i]
    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        pts = data['prob']

        pred = {}
        pred.update({"pts": pts})

        # save keypoints
        path_to_pts = Path(path_pts, "{}.npz".format(f_num))
        np.savez_compressed(path_to_pts, **pred)

        pass


if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    extract(args)
