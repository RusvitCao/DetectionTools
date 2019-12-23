import mmcv
import os
import numpy as np
import copy


def fusion(input_pkls_dir, outfile, iou_thr, in_score_thr, out_score_thr):

    pkl_names_list = os.listdir(input_pkls_dir)
    num_pkls = len(pkl_names_list)
    if pkl_names_list == []:
        raise ValueError('Invalid input dir: can`t be NULL!')
    scale_factors = [float(pkl_name[pkl_name.index('_')+1:pkl_name.index('pkl')-1]) 
                          for pkl_name in pkl_names_list]

    # sort the fusion rank:
    # e.g. [0.5,0.75,1,1.5,2] -> [1,2,1.5,0.75,0.5]
    c = list(zip(scale_factors, pkl_names_list))
    c.sort()
    scale_factors, pkl_names_list = zip(*c)
    scale_factors = list(scale_factors)
    ind1 = scale_factors.index(1.0)
    scale_factors.append(scale_factors.pop(ind1))
    pkl_names_list = list(pkl_names_list)
    pkl_names_list.append(pkl_names_list.pop(ind1))
    scale_factors.reverse()
    pkl_names_list.reverse()
    print(scale_factors, pkl_names_list)

    # load detection results:
    pkls_list = [mmcv.load(input_pkls_dir + '/' + pkl_name) for pkl_name in pkl_names_list]
    all_imgs_det = []
    num_imgs = len(pkls_list[0])
    num_class = 20

    for i in range(num_imgs): # [ [], [], ..., [] ]
        img_det_fused = []
        img_dets_input = [pkl[i] for pkl in pkls_list]

        for ic in range(num_class): # [ arr, arr, ..., arr]
            inds_and_scrs = [[[ _, det[ic][_, -1]] for _ in range(np.shape(det[ic])[0])]
                                for det in img_dets_input] 
            for idet in range(len(img_dets_input)):
                if inds_and_scrs[idet] != []: # [] no need to sort.
                    inds_and_scrs[idet].sort(key=lambda x: x[1]) # sort based on scores

            cls_det = []
            # print(len(img_dets_input))
            for idet in range(len(img_dets_input)):
                # print(len(img_dets_input[idet]))
                while inds_and_scrs[idet] != []:
                    base_det_info = inds_and_scrs[idet][-1]
                    if base_det_info[1] < in_score_thr: break # scores too small, break.
                    instance_det = []
                    base_det = img_dets_input[idet][ic][base_det_info[0], :]
                    base_scale = scale_factors[idet]
                    base_det = base_det / base_scale
                    if np.sum(base_det) > 1e4 or np.sum(base_det) < 0:
                        print(base_scale, base_det)
                    for jdet in range(idet+1, len(img_dets_input)):
                        instance_ind = len(inds_and_scrs[jdet])-1
                        _exist = False
                        while (instance_ind >= 0 and 
                              inds_and_scrs[jdet][instance_ind][0] >= in_score_thr):
                            cur_det = img_dets_input[jdet][ic][inds_and_scrs[jdet][instance_ind][0],:]
                            cur_scale = scale_factors[jdet]
                            cur_det_copy = copy.deepcopy(cur_det)
                            cur_det_copy[:-1] = cur_det_copy[:-1] / cur_scale
                            iou = iou_cal(cur_det, base_det) # `iou_cal`
                            if iou >= iou_thr: 
                                _exist = True
                                break
                            else: instance_ind -= 1
                        if _exist:
                            instance_det.append(cur_det_copy)
                            inds_and_scrs[jdet].pop(instance_ind)

                    instance_det.append(base_det)
                    inds_and_scrs[idet].pop()
                    instance_fusion = fuse(instance_det, num_pkls) # `fuse` 
                    cls_det.append(instance_fusion)
            
            if cls_det == []: # no bbox for this cls, return a null array[0, 5]
                cls_fusion = np.zeros([0,5], dtype=np.float32)
            else: 
                cls_fusion = []
                for cdet in cls_det:
                    if cdet[-1] >= out_score_thr: cls_fusion.append(cdet)
                if cls_fusion != []:
                    cls_fusion = np.stack(cls_fusion)
                else: cls_fusion = np.zeros([0,5], dtype=np.float32)
            img_det_fused.append(cls_fusion)

        # print('-----------------------')
        # print(img_det_fused) # print fusion result on each img.
        all_imgs_det.append(img_det_fused)

    # with open(out_file, 'w') as f:
    mmcv.dump(all_imgs_det, outfile)

    return all_imgs_det        
        

# def iou_cal(bbox1, bbox2, scale1, scale2):
def iou_cal(bbox1, bbox2): 
    # b1 = bbox1[:-1] / scale1
    # b2 = bbox2[:-1] / scale2
    b1 = bbox1[:-1]
    b2 = bbox2[:-1]
    # b: [x1, y1, x2, y2]
    ov_w = min(b1[2], b2[2]) - max(b1[0], b2[0])
    ov_h = min(b1[3], b2[3]) - max(b1[1], b2[1])
    if ov_w <= 0 or ov_h <= 0:
        return 0
    ov = ov_w * ov_h
    aa = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) 
    return ov / (aa - ov)



# fuse1: return box is the base box.
def fuse(bboxes, num_dets):
    res = np.zeros([5], dtype=np.float32)
    res[:-1] = bboxes[-1][:-1]
    sum_scr = 0
    for bbox in bboxes:
        sum_scr += bbox[-1]
    res[-1] = sum_scr / num_dets
    return res


# fuse2: return box with average cords.
def fuse2(bboxes, num_dets):   
    num_bbox = len(bboxes)
    res = np.zeros([5], dtype=np.float32) 
    for bbox in bboxes:
        res += bbox
    res[-1] = res[-1] / num_dets
    res[:-1] = res[:-1] / num_bbox
    return res


if  __name__ == '__main__':
    input_dir = '/home/caohaotian/WorkSpace/temp/multi_inference_results'
    output_pkl = '/home/caohaotian/WorkSpace/temp/multi_infe_fusion_res/fused_results__gen_iou0_70_scr0_00.pkl'
    iou_thr = 0.7
    in_score_thr = 0.0
    out_score_thr = 0.0
    fusion(input_dir, output_pkl, iou_thr, in_score_thr, out_score_thr)



