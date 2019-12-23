import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils


def merge_pkls(input_pkls_dir, outfile, 
               iou_thr, score_thr, mask_thr):

    pkl_names_list = os.listdir(input_pkls_dir)
    num_pkls = len(pkl_names_list)
    if pkl_names_list == []:
        raise ValueError('Invalid input dir: can`t be NULL!')

    # load detection results:
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

            # merge dets of class(ic).
            cls_dets_bbox_merged, cls_dets_segs_merged = bbox_merge(cls_dets_bbox, cls_dets_segs, iou_thr, score_thr, mask_thr)   

            img_dets_bbox_merged.append(cls_dets_bbox_merged)
            img_dets_segs_merged.append(cls_dets_segs_merged)
 
        all_imgs_det.append((img_dets_bbox_merged, img_dets_segs_merged))
    mmcv.dump(all_imgs_det, outfile)

        
def bbox_merge(dets, segs, iou_thr, scr_thr, mask_thr):
    # dets: [[x1, y1, x2, y2, score], ... ]
    if dets.shape[0] <= 1:
        return dets, segs
    order = dets[:, -1].ravel().argsort()[::-1]
    dets = dets[order, :]
    scr_keep_inds = (np.where(dets[:, -1] > scr_thr))[0]
    dets = dets[scr_keep_inds, :]
    segs = [segs[ind] for ind in scr_keep_inds]

    dets_res = np.zeros([0, 5])
    segs_res = []
    imgHeight, imgWidth = 1024, 2048

    while dets.shape[0] > 0:
        num = dets.shape[0]
        # IoU
        area = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
        xx1 = np.maximum(dets[0, 0], dets[:, 0])
        yy1 = np.maximum(dets[0, 1], dets[:, 1])
        xx2 = np.minimum(dets[0, 2], dets[:, 2])
        yy2 = np.minimum(dets[0, 3], dets[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_inds = np.where(o >= iou_thr)[0]
        dets_to_merge = dets[merge_inds, :]
        segs_to_merge = [segs[ind] for ind in merge_inds]
        dets = np.delete(dets, merge_inds, 0) # remained dets and segs after remerge.
        segs = [segs[i] for i in range(num) if i not in merge_inds]

        if merge_inds.shape[0] <= 1:
            dets_res = np.row_stack((dets_res, dets_to_merge))
            segs_res += segs_to_merge

        else:
            scores = dets_to_merge[:, -1:]
            dets_to_merge[:, :-1] = dets_to_merge[:, :-1] * np.tile(scores, (1, 4))
            max_score = np.max(scores)
            det_merged = np.zeros((1, 5))
            det_merged[:, :-1] = np.sum(dets_to_merge[:, :-1], axis=0) / np.sum(scores)
            det_merged[:, -1] = max_score
            dets_res = np.row_stack((dets_res, det_merged))

            img = np.zeros((imgHeight, imgWidth))
            for i in range(merge_inds.shape[0]):
                mask = maskUtils.decode(segs_to_merge[i]).astype(np.bool)
                img[mask] += scores[i, -1]
            img = img / np.max(img)
            img[img >= mask_thr] = 1
            img[img < mask_thr]  = 0
            img = img.astype(np.uint8)
            # print(img.shape)
            seg_merged = maskUtils.encode(np.array(img[:, :, np.newaxis], order='F'))[0]
            segs_res.append(seg_merged)

    return dets_res, segs_res


if  __name__ == '__main__':
    # input_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4'
    # output_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4_mres/merge_dets__merge_scr0_00_iou0_50.pkl'

    input_dir = '/home/cao/workspace/PASCAL_VOC/Utils/pkls_to_merge'
    output_pkl = input_dir + '/merge_dets__scr0_05__iou0_50.pkl'

    iou_thr = 0.8
    score_thr = 0.05
    mask_thr = 0.5
    merge_pkls(input_dir, output_pkl, iou_thr, score_thr, mask_thr)



