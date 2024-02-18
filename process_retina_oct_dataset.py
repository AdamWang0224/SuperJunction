import numpy as np
import cv2
import os

input_dir = './datasets/oct_original_pairs_52/'
output_dir = './datasets/retina_test/img/'
for i in range(1, 53):
    path_to_input = input_dir + str(i) + '/'
    files = os.listdir(path_to_input)
    for file in files:
        if '.jpg' in file:
            path_to_input_img = path_to_input + file
            img = cv2.imread(path_to_input_img)

            height = img.shape[0]
            width = img.shape[1]
            left = int(width/2) - int(height/2)
            right = left + height

            img_crop = img[:, left:right, :]
            path_to_output_img = output_dir + str(i) + '.jpg'
            cv2.imwrite(path_to_output_img, img_crop)
