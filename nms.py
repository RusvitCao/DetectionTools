# used for nms-op over a set of dense dets.
# conclude 2: nms and soft-nms.
import os
import mmcv
import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4] # det scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # all areas of each bbox.
    order = scores.argsort()[::-1] # order based on scores.
    keep = []

    while order.size > 0:

        # order[0] represents the largest score det, keep it of course.
        i = order[0]
        keep.append(i)

        # xx1, xx2, xx3, xx4 are det[i] 
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # if w or h <= 0, overlap <- 0. 
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = w * h # 
        iou = overlap / (areas[i] + areas[order[1:]] - overlap)

        # inds represent indices between which and order[0], iou <= thresh. so keep`em.
        inds = np.where(iou <= thresh)[0]
        # order only keep `inds`, remove others.
        order = order[inds + 1]
    
    return keep


def py_cpu_softnms(boxes, method, iou_thr, scr_thr, sigma):
    N = boxes.shape[0]
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > iou_thr: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > iou_thr: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < scr_thr:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
    keep = [i for i in range(N)]
    return keep


"""
def py_cpu_softnms(dets, method, iou_thr, scr_thr, sigma):
    # py_cpu_softnms
    #     :param dets:   np array: [[x1, y1, x2, y2, scr]]
    #     :param method: 1. linear, 2. gauss
    #     :param iou_thr
    #     :param scr_thr
    #      :param sigma:  if method == 2, sigma is for gauss-dist func.
    #     :return:       keep inds.

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        # find the maxscore in [i+1:].
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore: # swap op
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:   # linear
            weight = np.ones(ovr.shape)
            weight[ovr > iou_thr] = weight[ovr > iou_thr] - ovr[ovr > iou_thr]
        elif method == 2: # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:             # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > iou_thr] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    # inds = dets[:, 4][scores > scr_thr]
    inds = (np.where(scores > scr_thr))[0]
    keep = inds.astype(int)
    # print(keep)
    return keep
"""


def catagory_nms_op(img_dets, iou_thresh):
    """multi-class nms
    """
    res = []
    for cls_det in img_dets: 
        if cls_det.shape[0] > 0:
            keep_inds = py_cpu_nms(cls_det, iou_thresh) 
            res_cls_det = cls_det[keep_inds, :]
            res.append(res_cls_det)
        else: res.append(cls_det)
    return res
    

def catagory_softnms_op(img_dets, iou_thresh,
                       score_thresh, soft_nms_method, sigma):
    res = []
    for cls_det in img_dets:
        if cls_det.shape[0] > 0:
            keep_inds = py_cpu_softnms(cls_det, soft_nms_method, 
                         iou_thresh, score_thresh, sigma)
            res_cls_det = cls_det[keep_inds, :]
            res.append(res_cls_det)
        else: res.append(cls_det)
    return res 


def all_dataset_nms_op(pkl_dets=None, 
                       out_file=None,
                       soft_nms=False, 
                       iou_thresh=0.5,
                       score_thresh=0.3,
                       soft_nms_method=2,
                       sigma=1):
    pkl_list = mmcv.load(pkl_dets)
    for i, img_dets in enumerate(pkl_list):
        if soft_nms:
            pkl_list[i] = catagory_softnms_op(img_dets, iou_thresh,
                             score_thresh, soft_nms_method, sigma)
        else:
            pkl_list[i] = catagory_nms_op(img_dets, iou_thresh)
    mmcv.dump(pkl_list, out_file)


def main():
    # pkl_dets = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4_mres/merge_dets__nms_merge_scr0_01_iou0_50.pkl'
    # out_file = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Cascade_x101_yzz/work_dirs/mtest_results_epoch4_mres/nms_merge_dets__merge_scr0_00_iou0_50__nms_hard_iou0_50_scr0_00.pkl'

    pkl_dets = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/mtest_results_epoch5_mres/merge_dets__merge_scr0_00_iou0_50.pkl'
    out_file = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/HTC_x101_VOCaug/work_dirs/mtest_results_epoch5_mres/nms_merge_dets__merge_scr0_00_iou0_50__nms_hard_iou0_50_scr0_00.pkl'

    iou_thresh = 0.5
    # soft_nms = False
    # score_thresh = 0.3
    # sigma = 1
    # soft_nms_method = 2
    all_dataset_nms_op(pkl_dets,
                       out_file,
                       iou_thresh=iou_thresh)



if __name__ == '__main__':
    main()


    
