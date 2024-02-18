import os

input_dir = './datasets/retina_test/img/'
files = os.listdir(input_dir)

count = 0
for file in files:
    file_type = file[-4:]
    path_to_file = input_dir + file
    os.rename(path_to_file, input_dir + str(count) + file_type)
    count += 1
