import matplotlib.pyplot as plt
import numpy as np
import os

# confidence_lst = ['0%', '2.5%', '5%', '7.5%', '10%', '12.5%', '15%', '17.5%', '20%']
confidence_lst = ['0%', '5%', '10%', '15%', '20%']
pixel_threshold = 5
# epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
specificity_confidence_dict = {}
for confidence in confidence_lst:
    specificity_epoch_dict = {}
    for epoch in epochs:
        path_to_ground_truth_pts = './datasets/retina_test/pts/'
        path_to_detected_pts = './logs/superpoint_retina_test/predictions_' + str(epoch) + '_' + confidence + '/'

        # computation of specificity
        print(f'confidence level = {confidence}')
        print(f'epoch = {epoch}')
        files_dt_pts = os.listdir(path_to_detected_pts)
        files_dt_pts.sort(key=lambda x: int(x[:-4]))
        specificity_lst = []
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
        avg_specificity = np.average(specificity_lst)
        specificity_epoch_dict[epoch] = avg_specificity

    specificity_confidence_dict[confidence] = specificity_epoch_dict

print(specificity_confidence_dict)

for confidence in confidence_lst:
    plt.plot(specificity_confidence_dict[confidence].keys(),
             specificity_confidence_dict[confidence].values(), label=confidence)
plt.legend()
plt.title('Specificity vs. Epoch')
plt.savefig('specificity_vs_epoch.png')
# plt.show()
