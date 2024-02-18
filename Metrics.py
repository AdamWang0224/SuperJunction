# *coding:utf-8 *

import sklearn.metrics as metrics
import numpy as np
import cv2


def calculate_auc_test(prediction, label):
    # read images

    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()
    label_1D = label_1D / 255

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))
    return auc


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return acc, sen, spe


def mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)  # I assume this is faster as mask1 == 1 is a bool array
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def dice(im1, im2, empty_score=1.0):
    """
    This code is from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


if __name__ == '__main__':
    print("Testing the dice coefficient")
    a = np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]])

    b = np.array([[0, 0, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 0, 0]])

    dice_coe = dice(a, b)
    print("dice is ", dice_coe)

    mask_iou = mask_iou(a, b)
    print("mask iou is ", mask_iou)
