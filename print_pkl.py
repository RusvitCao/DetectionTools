import mmcv
import numpy as np

input_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/mtest_results_epoch1/result800.pkl'
pkl = mmcv.load(input_pkl)
# img_list = '/home/store-1-img/caohaotian/wujiahong_for_cao/split_test.txt'

"""
img_names = []
for line in open(img_list):
    img_names.append(line[:-1])

"""

for i in range(2):
    cur = pkl[i]
    for j in range(len(cur)):
        for k in range(cur[j].shape[0]):
            cur_inst = cur[j][k, :]
            print(cur_inst)
        print('-----------') 

"""
for i in range(len(pkl)):
    if img_names[i] == '2008_006758':
        cur = pkl[i] 
        for j in range(len(cur)): 
            for k in range(cur[j].shape[0]):
                cur_inst = cur[j][k, :] 
                print(cur_inst)
            print('-----------')
        break
"""
