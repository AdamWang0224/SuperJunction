import numpy as np
import os

path_to_ground_truth = './datasets/retina_test/pts/'
path_to_prediction = './logs/superpoint_retina/predictions/'

files_ground_truth = os.listdir(path_to_ground_truth)

precision_lst = []
recall_lst = []
for file in files_ground_truth:
    data_ground_truth = np.load(path_to_ground_truth + file)
    data_prediction = np.load(path_to_prediction + file)

    pts_ground_truth = data_ground_truth['pts']
    pts_prediction = data_prediction['prob']

    threshold = 0.015
    pts_prediction = pts_prediction[pts_prediction[:, 2] >= threshold]

    print('Number of detected pts: ', len(pts_prediction))
    print('Number of ground truth pts list: ', len(pts_ground_truth))

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
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

    precision_lst.append(precision)
    recall_lst.append(recall)

print('Average precision: ', np.average(precision_lst))
print('Average recall: ', np.average(recall_lst))

