import os
import mmcv
import numpy as np
from mmdet.core import get_classes
import pycocotools.mask as maskUtils


def show_result(img, result, dataset='voc', score_thr=0.3, out_file=None): # dataset:[coco, voc, ...]
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None: # show the segmentation.
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0] # select boxes whose score higher than thr.
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)       # RGB-(1,3)
            mask = maskUtils.decode(segms[i]).astype(np.bool) # the mask predicted.
            img[mask] = img[mask] * 0.5 + color_mask * 0.5    # color fusion to make it high-lighted!
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=False,
        out_file=out_file)


def parse_pkl(preds):
    """
    for c in range(len(preds)):
        cls_pred = preds[c]
        if np.shape(cls_pred)[0] > 0:
            for i in np.shape(cls_pred)[0]:
                bbox = []
                # bbox cords:
                for j in range(4):
                    bbox.append(cls_pred[i, j])
                # bbox scores:
                bbox.append(cls_pred[i, -1])
                # bbox catogary:
                bbox.append(c+1) """
    return preds


def main():

    input_pkl = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Single_Infs/res_pkls/result800.pkl'

    img_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Single_Infs/source_imgs'

    out_dir = '/home/cao/workspace/PASCAL_VOC/VOC_Tasks/Single_Infs/res_imgs'
    show_interval = 100

    if not os.path.exists(out_dir): os.mkdir(out_dir)
    pkl = mmcv.load(input_pkl)
    img_names_list = os.listdir(img_dir)
    print('num_imgs: ', len(img_names_list))
    print('num_dets: ', len(pkl))

    for i in range(len(pkl)):
        result = parse_pkl(pkl[i])
        show_result(img_dir + '/' + img_names_list[i], 
                    result, dataset='voc',
                    score_thr=0.3, out_file=out_dir + '/' + img_names_list[i])


if __name__ == '__main__':
    main()










