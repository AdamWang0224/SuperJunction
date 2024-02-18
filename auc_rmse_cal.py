# extract keypoints, matching
# split matching keypoints to partitions
# This code runs the trained model (superpoint/superglue) to extract keypoints

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
import pickle
from SuperGlue_models.matching import Matching
from SuperGlue_utils.common import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc,
                          read_image, read_image_with_homography, read_image_pair,
                          rotate_intrinsics, rotate_pose_inplane, compute_pixel_error,
                          scale_intrinsics, weights_mapping, download_base_files, download_test_images)
from SuperGlue_utils.preprocess_utils import torch_find_matches

# 补充宏包
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression #导入线性回归模块
from sklearn.preprocessing import PolynomialFeatures

torch.set_grad_enabled(False)


def compute_group_id(range_group, point_mat):
    point_x = point_mat[0, 0]
    point_y = point_mat[0, 1]
    # compute group center
    group_center_dict = {}
    for key, value in range_group.items():
        x_min = value[0][0]
        x_max = value[0][1]
        y_min = value[1][0]
        y_max = value[1][1]
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        group_center_dict[key] = (center_x, center_y)

    min_dist = 99999999
    best_group_id = 0
    for group_id in range(len(group_center_dict)):
        center_x = group_center_dict[group_id][0]
        center_y = group_center_dict[group_id][1]
        distance = (point_x - center_x) ** 2 + (point_y - center_y) ** 2
        if distance < min_dist:
            min_dist = distance
            best_group_id = group_id

    return best_group_id


def compute_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    # assert (len(p_error) == 49)
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}


if __name__ == '__main__':
    num_partition = 2
    h_threshold = 300
    original_img_size = [2912, 2912]
    resized_img_size = [1024, 1024]
    scale_x = original_img_size[0] / resized_img_size[0]
    scale_y = original_img_size[1] / resized_img_size[1]

    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_homography', type=str, default='assets/fire_sample_pairs.txt',
        help='Path to the list of image pairs and corresponding homographies')
    parser.add_argument(
        '--input_dir', type=str, default='assets/fire_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_fire_pairs_partition_test/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1024, 1024],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', default='retina',  # retina
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--min_matches', type=int, default=12,
                        help="Minimum matches required for considering matching")
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--viz', action='store_true', default=True,
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_homography, 'r') as f:
        homo_pairs = f.readlines()

    if opt.max_length > -1:
        homo_pairs = homo_pairs[0:np.min([len(homo_pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(homo_pairs)
    download_base_files()
    download_test_images()

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    try:
        curr_weights_path = str(weights_mapping[opt.superglue])
    except:
        if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
            curr_weights_path = str(opt.superglue)
        else:
            raise ValueError("Given --superglue path doesn't exist or invalid")

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights_path': curr_weights_path,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)

    rmse_lst_A = []
    rmse_lst_P = []
    rmse_lst_S = []

    rmse_lst_A_new = []
    rmse_lst_P_new = []
    rmse_lst_S_new = []

    inaccurate = 0
    acceptable = 0

    inaccurate_new = 0
    acceptable_new = 0
    
    img_num = len(homo_pairs) - 1


    sum_of_error = 0
    sum_of_new_error = 0

    abnomarl_value_x = 0
    abnomarl_value_y = 0

    sum_of_rmse = 0
    sum_of_rmse_S = 0
    sum_of_rmse_A = 0
    sum_of_rmse_P = 0

    numS = 0
    numA = 0
    numP = 0
    
    num_hyb_S = 0
    num_hyb_A = 0
    num_hyb_P = 0

    for i, info in enumerate(homo_pairs):

        split_info = info.strip().split(' ')

        image0_name = split_info[0]
        image1_name = split_info[1]

        print('===================')
        print('Working on image pair: ', image0_name[:3])

        if 'P37' in image0_name:
            continue
        # homo_info = list(map(lambda x: float(x), split_info[1:]))
        # homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
        stem0 = Path(image0_name).stem
        stem1 = Path(image1_name).stem
        stem = stem0 + '_' + stem1
        matches_path = output_dir / '{}_matches.npz'.format(stem)
        eval_path = output_dir / '{}_evaluation.npz'.format(stem)
        viz_path = output_dir / '{}_matches.{}'.format(stem, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_evaluation.{}'.format(stem, opt.viz_extension)


        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))
            continue

        image0, image1, inp0, inp1 = read_image_pair(input_dir / image0_name, input_dir / image1_name, device,
                                                     opt.resize, opt.resize_float)

        if image0 is None or image1 is None:
            print('Problem reading image pair: {} and {}'.format(
                input_dir / image0_name, input_dir / image1_name))
            exit(1)
        timer.update('load_image')

        # Modify the image to show in the visualization part
        image0 = image0.astype(np.uint8)
        image1 = image1.astype(np.uint8)
        image0_bgr = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
        image1_bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

        

        # Perform the matching
        # extract keypoints
        pred, data = matching.forward_superpoint({'image0': inp0, 'image1': inp1})
        # do global matching
        pred = matching.forward_superglue(pred, data)

        # save file
        pred_ = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred_['keypoints0'], pred_['keypoints1']
        matches, conf = pred_['matches0'], pred_['matching_scores0']

        # desc0, desc1 = pred_['descriptors0'], pred_['descriptors1']

        # out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
        #                'matches': matches, 'match_confidence': conf,
        #                'descriptors0': desc0, 'descriptors1': desc1}
        #
        # np.savez(str(matches_path), **out_matches)

        # for visualization
        valid = matches > -1  # match=-1意味着匹配不成功
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        color = cm.jet(mconf)

        # 放弃颜色展示，将所有线段全部调成红色
        vec = np.array([0, 0.9, 0.1, 0.6])
        color[:,:] = vec
        
        text = [
            'SuperJunction',
            # 'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        text_null = []

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem0),
        ]

        make_matching_plot(
            image0_bgr, image1_bgr, kpts0, kpts1, mkpts0, mkpts1, color,
            text_null, viz_path, opt.show_keypoints,
            opt.fast_viz, opt.opencv_display, 'Matches', text) # 第一个text_null参数位为左上角大文本位，第二个text参数位为右下角小文本位

        timer.update('viz_match')
        timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))

        # for evaluation
        matching_kpts0 = []
        matching_kpts1 = []

        for match_id, match in enumerate(matches):
            if match != -1:
                kpts0_x = kpts0[match_id][0] * scale_x
                kpts0_y = kpts0[match_id][1] * scale_y
                kpts1_x = kpts1[match][0] * scale_x
                kpts1_y = kpts1[match][1] * scale_y

                matching_kpts0.append([kpts0_x, kpts0_y])
                matching_kpts1.append([kpts1_x, kpts1_y])

        matching_kpts0_mat = np.array(matching_kpts0)
        matching_kpts1_mat = np.array(matching_kpts1)

        # compute global homography
        h, mask = cv2.findHomography(matching_kpts0_mat, matching_kpts1_mat, cv2.LMEDS)

        # f_X = np.polyfit(matching_kpts0_mat[:, 0], matching_kpts1_mat[:, 0], 4)        # 用5次多项式拟合
        # f_Y = np.polyfit(matching_kpts0_mat[:, 1], matching_kpts1_mat[:, 1], 3)        # 用5次多项式拟合
        # popt_X, pcov_X = curve_fit(PolyFunc, matching_kpts0_mat[:,0:1], matching_kpts1_mat[:,0])                # 曲线拟合，popt为函数的参数list
        # popt_Y, pcov_Y = curve_fit(PolyFunc, matching_kpts0_mat[:,0:1], matching_kpts1_mat[:,1])                 # 曲线拟合，popt为函数的参数list
        matching_kpts0_mat_list = matching_kpts0_mat.tolist()
        for index in range(2,3):
            data_X = pd.DataFrame({'IN':matching_kpts0_mat_list, 'OUT':matching_kpts1_mat[:,0]})
            data_train_X = np.array(data_X['IN']).reshape(data_X['IN'].shape[0],1)
            data_test_X = data_X['OUT']
	
            poly_reg_X = PolynomialFeatures(degree = index) 
            X_ploy = poly_reg_X.fit_transform(matching_kpts0_mat_list)
            regr_X = LinearRegression()
            regr_X.fit(X_ploy,data_test_X)
            

            if(regr_X.score(X_ploy,data_test_X) >= 0.99):
                break

            
        # print("vvvvvvvvvvvvvvvvvvvvvvvvv  Here is X  vvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print("degree = ", index)
        # print("coefficients = ", regr_X.coef_)
        # print("intercept = ", regr_X.intercept_)
        # print("R^2 = ",regr_X.score(X_ploy,data_test_X))								
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^  End of X  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        

        for index in range(2,3):
            data_Y = pd.DataFrame({'IN':matching_kpts0_mat_list, 'OUT':matching_kpts1_mat[:,1]})
            data_train_Y = np.array(data_Y['IN']).reshape(data_Y['IN'].shape[0],1)				
            data_test_Y = data_Y['OUT']
	
            poly_reg_Y = PolynomialFeatures(degree = index)
            Y_ploy = poly_reg_Y.fit_transform(matching_kpts0_mat_list)
            regr_Y = LinearRegression()
            regr_Y.fit(Y_ploy,data_test_Y)
        
            if(regr_Y.score(Y_ploy,data_test_Y) >= 0.99):
                break

        # print("vvvvvvvvvvvvvvvvvvvvvvvvv  Here is Y  vvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print("degree = ", index)
        # print("coefficients = ", regr_Y.coef_)
        # print("intercept = ", regr_Y.intercept_)
        # print("R^2 = ",regr_Y.score(Y_ploy,data_test_Y))
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^  End of Y  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                


        h_abs = abs(h[0, 2]) + abs(h[1, 2])

        # manipulate with data to partition keypoints
        x_min = np.floor(min(matching_kpts0_mat[:, 0]))
        x_max = np.ceil(max(matching_kpts0_mat[:, 0]))
        y_min = np.floor(min(matching_kpts0_mat[:, 1]))
        y_max = np.ceil(max(matching_kpts0_mat[:, 1]))

        x_interval = np.ceil((x_max - x_min) / num_partition)
        y_interval = np.ceil((y_max - y_min) / num_partition)

        range_group = {}
        for partition_id in range(num_partition ** 2):
            range_group[partition_id] = []
        for i_ in range(num_partition):
            x_range_start = x_min + i_ * x_interval
            if i_ == num_partition - 1:
                x_range_end = x_range_start + x_interval
            else:
                x_range_end = x_range_start + x_interval - 1
            for j_ in range(num_partition):
                y_range_start = y_min + j_ * y_interval
                if j_ == num_partition - 1:
                    y_range_end = y_range_start + y_interval
                else:
                    y_range_end = y_range_start + y_interval - 1
                group_id = i_ * num_partition + j_
                range_group[group_id].append((x_range_start, x_range_end))
                range_group[group_id].append((y_range_start, y_range_end))

        # partition matching keypoints to groups
        kpts0_group = {}
        kpts1_group = {}
        for group_id in range(num_partition ** 2):
            kpts0_group[group_id] = []
            kpts1_group[group_id] = []

        for idx in range(matching_kpts0_mat.shape[0]):
            kpt0 = matching_kpts0_mat[idx]
            kpt1 = matching_kpts1_mat[idx]
            for group_id in range(num_partition ** 2):
                x_start = range_group[group_id][0][0]
                x_end = range_group[group_id][0][1]
                y_start = range_group[group_id][1][0]
                y_end = range_group[group_id][1][1]
                if (kpt0[0] >= x_start) and (kpt0[0] <= x_end) and (kpt0[1] >= y_start) and (kpt0[1] <= y_end):
                    kpts0_group[group_id].append(kpt0)
                    kpts1_group[group_id].append(kpt1)

        # compute local homographies
        h_dict = {}
        for group_id in range(num_partition ** 2):
            if len(kpts0_group[group_id]) < 6:
                h_dict[group_id] = h
            else:
                primary = np.array(kpts0_group[group_id])
                secondary = np.array(kpts1_group[group_id])
                h_local, mask = cv2.findHomography(primary, secondary, cv2.LMEDS)
                h_dict[group_id] = h_local

        # load ground truth file
        f_gt = open('./ground_truth/control_points_' + image0_name[:3] + '_1_2.txt')
        lines_gt = f_gt.read().splitlines()
        f_gt.close()


        transformed_primary_row_mat_list = []
        secondary_row_mat_list = []

        polynomial_mat_list = []

        for line_gt in lines_gt:
            primary_row = []
            secondary_row = []
            coords = line_gt.split(' ')

            # 将Ground Truth从txt文件中提取成np.array

            primary_row.append(float(coords[0]))
            primary_row.append(float(coords[1]))
            secondary_row.append(float(coords[2]))
            secondary_row.append(float(coords[3]))

            primary_row_mat = np.array(primary_row)
            secondary_row_mat = np.array(secondary_row)

            # 转换形状 - length列

            primary_row_mat = np.reshape(primary_row_mat, (-1, len(primary_row_mat)))
            secondary_row_mat = np.reshape(secondary_row_mat, (-1, len(secondary_row_mat)))
            
            # compute group where the point is located
            group_id = compute_group_id(range_group, primary_row_mat)
            selected_h = h_dict[group_id]

            if h_abs >= h_threshold:
                new_data_poly_X = poly_reg_X.fit_transform(primary_row_mat)
                predicted_values_X = regr_X.predict(new_data_poly_X)

                new_data_poly_Y = poly_reg_Y.fit_transform(primary_row_mat)
                predicted_values_Y = regr_Y.predict(new_data_poly_Y)
                polynomial_matrix = np.column_stack((predicted_values_X, predicted_values_Y))

                transformed_primary_row_mat = polynomial_matrix

                if 'S' in image0_name:
                    num_hyb_S += 1
                elif 'A' in image0_name:
                    num_hyb_A += 1
                elif 'P' in image0_name:
                    num_hyb_P += 1
            else:
                transformed_primary_row_mat = cv2.perspectiveTransform(primary_row_mat.reshape(-1, 1, 2), h)
            # transformed_primary_row_mat = cv2.perspectiveTransform(primary_row_mat.reshape(-1, 1, 2), h)
            transformed_primary_row_mat = transformed_primary_row_mat.reshape(-1, 2)
            transformed_primary_row_mat_list.append(transformed_primary_row_mat)
            secondary_row_mat_list.append(secondary_row_mat)

            # polynomial_mat_list.append(polynomial_matrix)

        transformed_primary_mat = np.array(transformed_primary_row_mat_list).reshape(-1, 2)
        secondary_mat = np.array(secondary_row_mat_list).reshape(-1, 2)

        # polynomial_mat = np.array(polynomial_mat_list).reshape(-1, 2)

        error = secondary_mat - transformed_primary_mat
        # error_new = secondary_mat - polynomial_mat


        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!
        # STOP! DO NOT MOVE FURTHER!

        # compute RMSE
        rmse_lst = []
        for i in range(error.shape[0]):
            err = np.sqrt(np.square(error[i][0]) + np.square(error[i][1]))
            rmse_lst.append(err)
        rmse = np.average(rmse_lst)
        print('RMSE: ', rmse)

        if 'S' in image0_name:
            rmse_lst_S.append(rmse)           
        elif 'A' in image0_name:
            rmse_lst_A.append(rmse)    
        elif 'P' in image0_name:
            rmse_lst_P.append(rmse)

        # compute MAE
        mae_lst = []
        for i in range(error.shape[0]):
            err = np.sqrt(np.square(error[i][0]) + np.square(error[i][1]))
            mae_lst.append(err)
        mae = np.max(mae_lst)

        # compute MEE
        mee_lst = []
        for i in range(error.shape[0]):
            err = np.sqrt(np.square(error[i][0]) + np.square(error[i][1]))
            mee_lst.append(err)
        mee = np.median(mee_lst)

        if mae > 50 or mee > 20:
            inaccurate += 1
        else:
            acceptable += 1

            sum_of_rmse += rmse

            if 'S' in image0_name:
                numS += 1
                sum_of_rmse_S += rmse
            elif 'A' in image0_name:
                numA += 1
                sum_of_rmse_A += rmse
            elif 'P' in image0_name:
                numP += 1
                sum_of_rmse_P += rmse

        print('Inaccurate rate: ', inaccurate / img_num)
        print('Acceptable rate: ', acceptable / img_num)

    #     print("========================= Now FOR POLY ==================================")

    #     # compute RMSE
    #     rmse_lst_new = []
    #     for i in range(error_new.shape[0]):
    #         err_new = np.sqrt(np.square(error_new[i][0]) + np.square(error_new[i][1]))
    #         rmse_lst_new.append(err_new)
    #     rmse_new = np.average(rmse_lst_new)
    #     print('RMSE FOR POLY: ', rmse_new)

    #     if 'S' in image0_name:
    #         rmse_lst_S_new.append(rmse_new)
    #     elif 'A' in image0_name:
    #         rmse_lst_A_new.append(rmse_new)
    #     elif 'P' in image0_name:
    #         rmse_lst_P_new.append(rmse_new)

    #     # compute MAE
    #     mae_lst_new = []
    #     for i in range(error_new.shape[0]):
    #         err_new = np.sqrt(np.square(error_new[i][0]) + np.square(error_new[i][1]))
    #         mae_lst_new.append(err_new)
    #     mae_new = np.max(mae_lst_new)

    #     # compute MEE
    #     mee_lst_new = []
    #     for i in range(error_new.shape[0]):
    #         err_new = np.sqrt(np.square(error_new[i][0]) + np.square(error_new[i][1]))
    #         mee_lst_new.append(err_new)
    #     mee_new = np.median(mee_lst_new)

    #     if mae_new > 50 or mee_new > 20:
    #         inaccurate_new += 1
    #     else:
    #         acceptable_new += 1

    #     print('Inaccurate rate: ', inaccurate_new / img_num)
    #     print('Acceptable rate: ', acceptable_new / img_num)

    # print("Abnormal Value of X: ", abnomarl_value_x)
    # print("Abnormal Value of Y: ", abnomarl_value_y)

    result = compute_auc(rmse_lst_S, rmse_lst_P, rmse_lst_A)
    # result_new = compute_auc(rmse_lst_S_new, rmse_lst_P_new, rmse_lst_A_new)
    print('AUC S: ', result['s'])
    print('AUC P: ', result['p'])
    print('AUC A: ', result['a'])
    print('mAUC: ', result['mAUC'])
    
    final_img_num = numP+numA+numS
    print('final image number for S,A,P:', numS, numA, numP)
    print('average RMSE:', sum_of_rmse/final_img_num)
    print('average RMSE for S:', sum_of_rmse_S/numS)
    print('average RMSE for A:', sum_of_rmse_A/numA)
    print('average RMSE for P:', sum_of_rmse_P/numP)

    # print("hybrid matching:", num_hyb_S, num_hyb_A, num_hyb_P)
    
    # print('(POLY)AUC S: ', result_new['s'])
    # print('(POLY)AUC P: ', result_new['p'])
    # print('(POLY)AUC A: ', result_new['a'])
    # print('(POLY)mAUC: ', result_new['mAUC'])





