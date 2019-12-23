# transfer .pkl format det result to PASCAL-VOC submitting format.
# input: a .pkl
# output: a dir including 20 .txt representing each class` det results.

import os
import mmcv
import numpy as np
from mmdet.core import get_classes


def parse_pkldets(input_pkl, img_list, outdir, dataset='voc'):

    dets = mmcv.load(input_pkl)
    num_imgs = len(dets)
    num_class = 20
    VOC_CLASSES = get_classes(dataset)
    img_names_list = []    
    for line in open(img_list):
        img_names_list.append(line[:-1])

    if len(dets) != len(img_names_list):
        raise ValueError("Lengths don`t match!")

    prefix = outdir + '/' + 'comp4_det_test_'
    for ic in range(num_class):
        class_dets = []
        class_name = VOC_CLASSES[ic]
        for i, img_det in enumerate(dets):
            img_name = img_names_list[i]
            cls_dets = img_det[ic]
            if cls_dets.shape[0] <= 0:
                continue
            else:
                for j in range(cls_dets.shape[0]):
                    cur = [img_name,
                           "%.2f" % cls_dets[j][-1], 
                           str(int(cls_dets[j][0])),
                           str(int(cls_dets[j][1])),
                           str(int(cls_dets[j][2])),
                           str(int(cls_dets[j][3]))]
                    class_dets.append(cur)
        f = open(prefix + class_name + '.txt', 'w')
        for class_det in class_dets:
            class_det = ' '.join(class_det)
            f.writelines(class_det)
            f.write('\n')
        f.close()
        print(class_name, ' is done!')        


if __name__ == '__main__':
    # input_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/submit__ubuntuGA_HTC__all_Aug__flip__epoch3/mtest_results_epoch3/scale_of4__merge_dets__nms_merge_scr0_01_iou0_50.pkl'
    # out_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/submit__ubuntuGA_HTC__all_Aug__flip__epoch3/mtest_results_epoch3/Main/'

    # input_pkl = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_s204_bs8_lr5e-4_cocovoc_OrgVersion/work_dirs/mtest_results_epoch5_submit/merge_dets__nms_merge_scr0_01_iou0_50.pkl'
    # out_dir = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_s204_bs8_lr5e-4_cocovoc_OrgVersion/work_dirs/mtest_results_epoch5_submit/Main'

    input_pkl = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_withGA_s204_bs8_lr5e-4_cocovoc_OrgVersion/VOConly_ft_work_dirs/mtest_results_epoch3_submit_s800/merge_dets__nms_merge_scr0_01_iou0_50.pkl'
    out_dir = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_withGA_s204_bs8_lr5e-4_cocovoc_OrgVersion/VOConly_ft_work_dirs/mtest_results_epoch3_submit_s800/Main'

    # input_pkl = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_uP40_bs2_lr5e-4_cocovoc_libraVersion/work_dirs/mtest_results_epoch4_submit/merge_dets__nms_merge_scr0_01_iou0_50.pkl'
    # out_dir = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_uP40_bs2_lr5e-4_cocovoc_libraVersion/work_dirs/mtest_results_epoch4_submit/Main'

    img_list = '/home/cao/workspace/PASCAL_VOC/Dataset/VOCSubmitTest/VOCdevkit/VOC2012/ImageSets/VOC_Submit_Test.txt'
    parse_pkldets(input_pkl, img_list, out_dir)
 
                    
