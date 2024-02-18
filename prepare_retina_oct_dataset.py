import os
import shutil

# rename color fundus and oct images
# copy color fundus image to ./datasets/retina_test/img folder
input_dir = './datasets/oct_groudtruth_pairs_52/'
output_dir = './datasets/retina_test/img/'
for i in range(1, 53):
    path_to_img_dir = input_dir + str(i) + '/'
    files = os.listdir(path_to_img_dir)
    for file in files:
        file_type = file[-4:]
        if 'jpg' in file:
            path_to_img_cf = path_to_img_dir + file
            path_to_img_cf_new = path_to_img_dir + str(i) + file_type
            os.rename(path_to_img_cf, path_to_img_cf_new)
        elif 'bmp' in file:
            path_to_img_oct = path_to_img_dir + file
            path_to_img_oct_new = path_to_img_dir + str(i) + '_oct' + file_type
            os.rename(path_to_img_oct, path_to_img_oct_new)

for i in range(1, 53):
    path_to_src = input_dir + str(i) + '/' + str(i) + '.png'
    path_to_dst = output_dir + str(i) + '.png'
    shutil.copyfile(path_to_src, path_to_dst)


