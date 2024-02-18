import matplotlib.pyplot as plt
import numpy as np
import os

checkpoint = 45000
pixel_threshold = 5
confidence_thresholds = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
sensitivity_dict = {}
specificity_dict = {}
for confidence_threshold in confidence_thresholds:
    path_to_ground_truth_pts = './datasets/retina_test/pts/'
    path_to_detected_pts = './logs/superpoint_retina_test/predictions_' + str(checkpoint) + '_' + \
                           str(confidence_threshold) + '%/'

    # computation of sensitivity
    files_gt_pts = os.listdir(path_to_ground_truth_pts)
    files_gt_pts.sort(key=lambda x: int(x[:-4]))
    # matching_pts_count_for_files = []
    sensitivity_lst = []
    specificity_lst = []
    for file in files_gt_pts:
        filename = file[:-4]
        ground_truth_pts_dict = np.load(path_to_ground_truth_pts + file)
        ground_truth_pts = ground_truth_pts_dict['pts'].tolist()
        detected_pts_dict = np.load(path_to_detected_pts + filename + '.npz')
        detected_pts = detected_pts_dict['prob'].tolist()
        total_ground_truth_pts = len(ground_truth_pts)
        total_detected_pts = len(detected_pts)
        # loop over ground truth points and look for detected points
        total_matching_pts = 0
        for ground_truth_pt in ground_truth_pts:
            ground_truth_pt_x = ground_truth_pt[0]
            ground_truth_pt_y = ground_truth_pt[1]
            matching_pt_lst = []
            # check detected points located within range of ground truth point
            for detected_pt in detected_pts:
                detected_pt_x = detected_pt[0]
                detected_pt_y = detected_pt[1]
                if (abs(ground_truth_pt_x - detected_pt_x) <= pixel_threshold) and \
                        (abs(ground_truth_pt_y - detected_pt_y) <= pixel_threshold):
                    matching_pt_lst.append(detected_pt)
            if len(matching_pt_lst) > 1:
                best_distance = 999999
                for matching_pt in matching_pt_lst:
                    distance = (ground_truth_pt_x - matching_pt[0]) ** 2 + (ground_truth_pt_y - matching_pt[1]) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_pt = matching_pt
                total_matching_pts += 1
                detected_pts.remove(best_pt)
            elif len(matching_pt_lst) == 1:
                total_matching_pts += 1
                detected_pts.remove(matching_pt_lst[0])
        # matching_pts_count_for_files.append(total_matching_pts)
        if total_ground_truth_pts == 0:
            sensitivity = 0
        else:
            sensitivity = total_matching_pts/total_ground_truth_pts
        sensitivity_lst.append(sensitivity)

    # computation of specificity
    files_dt_pts = os.listdir(path_to_detected_pts)
    files_dt_pts.sort(key=lambda x: int(x[:-4]))
    for file in files_dt_pts:
        filename = file[:-4]
        detected_pts_dict = np.load(path_to_detected_pts + file)
        detected_pts = detected_pts_dict['prob'].tolist()
        ground_truth_pts_dict = np.load(path_to_ground_truth_pts + filename + '.npz')
        ground_truth_pts = ground_truth_pts_dict['pts'].tolist()
        total_detected_pts = len(detected_pts)
        total_ground_truth_pts = len(ground_truth_pts)
        total_matching_pts = 0
        for detected_pt in detected_pts:
            detected_pt_x = detected_pt[0]
            detected_pt_y = detected_pt[1]
            matching_pt_lst = []
            for ground_truth_pt in ground_truth_pts:
                ground_truth_pt_x = ground_truth_pt[0]
                ground_truth_pt_y = ground_truth_pt[1]
                if (abs(detected_pt_x - ground_truth_pt_x) <= pixel_threshold) and \
                        (abs(detected_pt_y - ground_truth_pt_y) <= pixel_threshold):
                    matching_pt_lst.append(ground_truth_pt)
            if len(matching_pt_lst) > 1:
                best_distance = 999999
                for matching_pt in matching_pt_lst:
                    distance = (detected_pt_x - matching_pt[0]) ** 2 + (detected_pt_y - matching_pt[1]) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_pt = matching_pt
                total_matching_pts += 1
                ground_truth_pts.remove(best_pt)
            elif len(matching_pt_lst) == 1:
                total_matching_pts += 1
                ground_truth_pts.remove(matching_pt_lst[0])
        # matching_pts_count_for_files.append(total_matching_pts)
        if total_detected_pts == 0:
            specificity = 0
        else:
            specificity = total_matching_pts / total_detected_pts
        specificity_lst.append(specificity)

    # compute average sensitivity and specificity
    avg_sensitivity = np.average(sensitivity_lst)
    avg_specificity = np.average(specificity_lst)

    sensitivity_dict[confidence_threshold] = avg_sensitivity
    specificity_dict[confidence_threshold] = avg_specificity

# plt.subplot(1, 2, 1)
# plt.plot(confidence_thresholds, sensitivity_dict.values(), marker='o')
#
# plt.subplot(1, 2, 2)
# plt.plot(confidence_thresholds, specificity_dict.values(), marker='*')
#
# plt.show()
print(sensitivity_dict)
print(specificity_dict)

fig, ax = plt.subplots(1, 2)

ax[0].plot(confidence_thresholds, sensitivity_dict.values(), marker='o')
ax[1].plot(confidence_thresholds, specificity_dict.values(), marker='*')

ax[0].set_title("Sensitivity vs. Confidence")
ax[1].set_title("Specificity vs. Confidence")

ax[0].set_xlabel('Confidence')
ax[0].set_ylabel('Sensitivity')
ax[1].set_xlabel('Confidence')
ax[1].set_ylabel('Specificity')

fig.suptitle('Checkpoint_' + str(checkpoint))
fig.tight_layout()
path_to_file = './logs/superpoint_retina_test/'
plt.savefig(path_to_file + 'Checkpoint_' + str(checkpoint) + '_sensitivity_specificity.png')
# plt.show()

# # create data
# t = np.arange(0., 5., 0.2)
#
# # plot with marker
# plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')