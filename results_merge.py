import os
import mmcv
import numpy as np


def merge_pkls(input_pkls_dir, outfile, 
               iou_thr, in_score_thr, out_score_thr,
               method='soft_merge'):
    pkl_names_list = os.listdir(input_pkls_dir)
    num_pkls = len(pkl_names_list)
    if pkl_names_list == []:
        raise ValueError('Invalid input dir: can`t be NULL!')

    # load detection results:
    pkls_list = [mmcv.load(input_pkls_dir + '/' + pkl_name) for pkl_name in pkl_names_list]
    all_imgs_det = []
    num_imgs = len(pkls_list[0])
    num_class = 20

    for i in range(num_imgs):
        img_dets_merged = []
        img_dets_input = [pkl[i] for pkl in pkls_list]

        for ic in range(num_class): # [ [arr1, arr2, ..., arr20] * num_pkls ]
            cls_dets_input = np.zeros([0, 5])
            for iscale in range(num_pkls):
                single_img_cls_dets_input = img_dets_input[iscale][ic]
                
                # stack dets of class(ic) from each pkls.
                if single_img_cls_dets_input.shape[0] > 0:
                    cls_dets_input = np.row_stack([cls_dets_input, single_img_cls_dets_input])
            
            # merge dets of class(ic).
            if 'soft' in method:
                cls_dets_merged = soft_bbox_merge(cls_dets_input, iou_thr, in_score_thr, out_score_thr)
            else:
                cls_dets_merged = bbox_merge(cls_dets_input, iou_thr, in_score_thr)   

            img_dets_merged.append(cls_dets_merged)
 
        all_imgs_det.append(img_dets_merged)
    mmcv.dump(all_imgs_det, outfile)

        
def bbox_merge(det, iou_thr, scr_thr):
    # det: [[x1, y1, x2, y2, score], ... ]
    if det.shape[0] <= 1:
        return det
    order = det[:, -1].ravel().argsort()[::-1]
    det = det[order, :]
    det = det[np.where(det[:, -1] > scr_thr)[0], :]

    dets = np.zeros([0, 5])
    while det.shape[0] > 0:
        # IoU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= iou_thr)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, :-1] = det_accu[:, :-1] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, -1])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, :-1] = np.sum(det_accu[:, :-1], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, -1] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets


def soft_bbox_merge(det, iou_thr, in_scr_thr, out_scr_thr):
    if det.shape[0] <= 1:
        return det
    order = det[:, -1].ravel().argsort()[::-1]
    det = det[order, :]
    det = det[np.where(det[:, -1] > in_scr_thr)[0], :]

    dets = np.zeros([0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= iou_thr)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, -1] = soft_det_accu[:, -1] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, -1] >= out_scr_thr)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, :-1] = det_accu[:, :-1] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, -1])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, :-1] = np.sum(det_accu[:, :-1], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, -1] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, -1].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


if  __name__ == '__main__':
    # input_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4'
    # output_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4_mres/merge_dets__merge_scr0_00_iou0_50.pkl'

    input_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/mtest_results_epoch5'
    output_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/mtest_results_epoch5_mres/merge_dets__merge_scr0_00_iou0_50.pkl'

    iou_thr = 0.5
    in_score_thr = 0.0
    out_score_thr = 0.0
    merge_pkls(input_dir, output_pkl, iou_thr, in_score_thr, out_score_thr)



