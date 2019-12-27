import os
import mmcv
import numpy as np
import json
import random


def merge_dets(input_jsons_dir, outfile, 
               iou_thr, in_score_thr, out_score_thr,
               method='hard_merge'):
    json_names_list = os.listdir(input_jsons_dir)
    num_jsons = len(json_names_list)
    if json_names_list == []:
        raise ValueError('Invalid input dir: can`t be NULL!')

    # load detection results:
    dics_list = []
    for json_name in json_names_list:
        with open(input_jsons_dir + '/' + json_name, 'r') as f:
            dics_list.append(json.load(f))
    all_imgs_det = {}
    num_imgs = len(dics_list[0])
    num_class = 1

    No = 1
    for imgName in dics_list[0]:
        print(No, imgName)
        No += 1

        if imgName == 'has_list': continue
        img_dets_merged = []
        img_dets_input = []

        for dic in dics_list:
            img_dets = [np.array(dic[imgName])]
            img_dets_input.append(img_dets)

        for ic in range(num_class):
            cls_dets_input = np.zeros([0, 5])

            for iscale in range(num_jsons):
                single_img_cls_dets_input = img_dets_input[iscale][ic]
                if single_img_cls_dets_input.shape[0] > 0:
                    cls_dets_input = np.row_stack([cls_dets_input, single_img_cls_dets_input])
            
            if 'soft' in method:
                cls_dets_merged = soft_bbox_merge(cls_dets_input, iou_thr, in_score_thr, out_score_thr)
            else:
                cls_dets_merged = bbox_merge(cls_dets_input, iou_thr, in_score_thr)
            img_dets_merged = cls_dets_merged.tolist()
 
        all_imgs_det[imgName] = img_dets_merged

    with open(outfile, 'w') as f:
        json.dump(all_imgs_det, f)

        
def bbox_merge(det, iou_thr, scr_thr):
    # det: [[x1, y1, x2, y2, score], ... ]
    if det.shape[0] <= 1:
        return det
    order = det[:, -1].ravel().argsort()[::-1]
    det = det[order, :]
    det = det[np.where(det[:, -1] > scr_thr)[0], :]

    dets = np.zeros([0, 5])

    wrongInds = []
    for i in range(det.shape[0]):
        if det[i,0] > det[i,2] or det[i,1] > det[i,3]:
            wrongInds.append(i)
    det = np.delete(det, np.array(wrongInds), 0)

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
    input_dir = '/media/cao/0EF30E3D0EF30E3D/TeMp/MOT_merge/jsons'
    output_json = '/media/cao/0EF30E3D0EF30E3D/TeMp/MOT_merge/result/mergedMOT.json'

    iou_thr = 0.5
    in_score_thr = 0.0
    out_score_thr = 0.0
    merge_dets(input_dir, output_json, iou_thr, in_score_thr, out_score_thr)



