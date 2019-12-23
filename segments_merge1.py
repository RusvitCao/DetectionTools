import os
import mmcv
import numpy as np
from nms import py_cpu_softnms
from nms import py_cpu_nms

def merge_pkls(input_pkls_dir, outfile, 
               nms_iou_thr, nms_score_thr, max_per_img):

    pkl_names_list = os.listdir(input_pkls_dir)
    num_pkls = len(pkl_names_list)
    if pkl_names_list == []:
        raise ValueError('Invalid input dir: can`t be NULL!')
    
    pkls_list = [mmcv.load(input_pkls_dir + '/' + pkl_name) for pkl_name in pkl_names_list]
    all_imgs_det = []
    num_imgs = len(pkls_list[0])
    num_class = 8

    for i in range(num_imgs):
        print(i)
        img_dets_bbox_merged, img_dets_segs_merged = [], []
        img_dets_input = [pkl[i] for pkl in pkls_list] # pkl[i]: (bbox_result, seg_result) for img(i).

        for ic in range(num_class): # [ [arr1, arr2, ..., arr20] * num_pkls ]
            cls_dets_bbox = np.zeros([0, 5])
            cls_dets_segs = []
            for iscale in range(num_pkls):
                single_img_cls_dets_bbox = img_dets_input[iscale][0][ic] # numpy array
                single_img_cls_dets_segs = img_dets_input[iscale][1][ic] # list
                
                # stack dets of class(ic) from each pkls.
                if single_img_cls_dets_bbox.shape[0] > 0:
                    cls_dets_bbox = np.row_stack([cls_dets_bbox, single_img_cls_dets_bbox])
                    cls_dets_segs += single_img_cls_dets_segs
             
            if cls_dets_bbox.shape[0] > 1:
                # nms_keep_inds = py_cpu_softnms(cls_dets_bbox, method=2, iou_thr=nms_iou_thr, scr_thr=nms_score_thr, sigma=0.5)
                nms_keep_inds = py_cpu_nms(cls_dets_bbox, thresh=nms_iou_thr)
                cls_dets_bbox_merged = cls_dets_bbox[nms_keep_inds, :]
                cls_dets_segs_merged = [cls_dets_segs[ind] for ind in nms_keep_inds]
            else: 
                cls_dets_bbox_merged = cls_dets_bbox
                cls_dets_segs_merged = cls_dets_segs
         
            # suppress low score bboxes.
            if cls_dets_bbox_merged.shape[0] > 1:
                scr_keep_inds = (np.where(cls_dets_bbox_merged[:, -1] > nms_score_thr))[0]
                # print(scr_keep_inds)
                cls_dets_bbox_merged = cls_dets_bbox_merged[scr_keep_inds, :] # if scr_keep_inds is NULL, return [] @shape(0, 5)
                cls_dets_segs_merged = [cls_dets_segs[ind] for ind in scr_keep_inds]
            
            img_dets_bbox_merged.append(cls_dets_bbox_merged)
            img_dets_segs_merged.append(cls_dets_segs_merged)
        scores = np.hstack([img_dets_bbox_merged[j][:, -1] for j in range(num_class)])
        if len(scores) > max_per_img:
            kth = len(scores) - max_per_img
            thresh = np.partition(scores, kth)[kth]

            for j in range(num_class):
                keep_inds = (np.where(img_dets_bbox_merged[j][:, -1] >= thresh))[0]
                # print(keep_inds)
                img_dets_bbox_merged[j] = img_dets_bbox_merged[j][keep_inds]
                img_dets_segs_merged[j] = [img_dets_segs_merged[j][ind] for ind in keep_inds]
        
        all_imgs_det.append((img_dets_bbox_merged, img_dets_segs_merged))
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
    # input_dir = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_uP40_bs2_lr5e-4_cocovoc_libraVersion/work_dirs/mtest_results_epoch4_submit'
    # output_pkl = '/media/cao/0EF30E3D0EF30E3D/WorkSpace/HTC_noGA_uP40_bs2_lr5e-4_cocovoc_libraVersion/work_dirs/mtest_results_epoch4_submit/merge_dets__nms_merge_scr0_01_iou0_50.pkl'

    input_dir = '/home/cao/workspace/PASCAL_VOC/Utils/pkls_to_merge'
    output_pkl = input_dir + '/merge_dets__scr0_05__iou0_50.pkl'

    nms_iou_thr = 0.5
    nms_score_thr = 0.05
    max_per_img = 100
    merge_pkls(input_dir, output_pkl, nms_iou_thr, nms_score_thr, max_per_img)



