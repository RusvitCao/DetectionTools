import os
import os.path
from shutil import copyfile

out_dir = '/home/caohaotian/WorkSpace/temp/temp_data/'
img_list = '/home/caohaotian/WorkSpace/Data/2012_Train_All_17125.txt'
source_dir = ['/home/caohaotian/DataSets/VOCdevkit/VOC2012/JPEGImages/', '/home/caohaotian/DataSets/VOCdevkit/VOC2012/Annotations/']

img_names_list = []
for line in open(img_list):
    img_names_list.append(line[:-1])

for i in range(len(img_names_list)):
    src_file1 = source_dir[0] + img_names_list[i] + '.jpg'
    tar_file1 = out_dir + 'imgs/' + img_names_list[i] + '.jpg'
    src_file2 = source_dir[1] + img_names_list[i] + '.xml'
    tar_file2 = out_dir + 'annos/' + img_names_list[i] + '.xml'
    if os.path.isfile(src_file1) and os.path.isfile(src_file2):
        copyfile(src_file1, tar_file1)
        copyfile(src_file2, tar_file2)
    else:
        print('Error!')

