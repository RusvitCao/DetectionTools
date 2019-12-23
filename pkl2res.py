import os
import cv2
import mmcv
import numpy as np
from mmdet.core import get_classes
import pycocotools.mask as maskUtils
import json

labelDic = {'person': 24, 'rider': 25, 'car': 26, 'truck': 27,
            'bus': 28, 'train': 31, 'motorcycle': 32, 'bicycle': 33}

def parse_pkldets(input_pkl, img_dir, outdir, score_thr=0.3, dataset='cityscapes'):
    dets = mmcv.load(input_pkl)
    num_imgs = len(dets)
    # num_class = 8
    VOC_CLASSES = get_classes(dataset)
    print(VOC_CLASSES)
    img_names_list = os.listdir(img_dir)

    for k in range(num_imgs):
        print(k)
        imgHeight, imgWidth = 1024, 2048
        imgName = img_names_list[k]
        
        # print(dets[k])
        bbox_result, segm_result = dets[k]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
 
        # labs, confs, polygons = [], [], []

        ftxt = open(outdir + imgName[:-4] + '.txt', 'w')
        # print(inds)
        for i in inds:
            img = np.zeros((imgHeight, imgWidth), np.uint8) 
            mask = maskUtils.decode(segms[i]).astype(np.bool) # mask predicted.

            # contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # try:
            #     contour = np.reshape(contour[0], [-1, 2])
            # except:
            #     print(k, i, contour, 'Segmentation Error!')
            #     continue 
            # polygon = []
            # for j in range(contour.shape[0]):
            #     polygon.append([int(contour[j, 0]), int(contour[j, 1])])
            # polygons.append(polygon)                        # append polygon of a instance.

            img[mask] = 255                                   # mask as white-color.
            cv2.imwrite(outdir + imgName[:-4] + str(i) + '.png', img)

            # confs.append(round(float(bboxes[i, -1]), 2))    # conf predicted.
            # labs.append(VOC_CLASSES[labels[i]])             # labels predicted.

            # print(imgName[:-4]+str(i)+'.png')
            # print(str(labelDic[VOC_CLASSES[labels[i]]]))
            ftxt.write(imgName[:-4]+str(i)+'.png'+' '+str(labelDic[VOC_CLASSES[labels[i]]])+' '+str(round(float(bboxes[i, -1]), 2))+'\n')

        # save_as_json(imgName, imgHeight, imgWidth, labs, confs, polygons, outdir)

"""      
def save_as_json(img_name, height, width, labels, confs, polygons, outdir):
    res = {"imgHeight": height, "imgWidth": width, "objects": []}
    for i in range(len(labels)):
        res["objects"].append({"label": labels[i], "score": confs[i], "polygon": polygons[i]})
    file_name = outdir + img_name[:-4] + ".json"
    with open(file_name, 'w') as file_obj:
        json.dump(res, file_obj)
"""
               
def main():

    root_dir = '/home/cao/workspace/PASCAL_VOC/Utils/pkls_to_merge/'
    # input_pkl = root_dir + 'merge_dets__scr0_05__iou0_50.pkl'    
    input_pkl = root_dir + 'results_htc_withGA_noCOCO_repeat8.pkl'

    img_dir = '/media/cao/0EF30E3D0EF30E3D/CityScape/dataset/test_all_city'
    out_dir = '/media/cao/0EF30E3D0EF30E3D/CityScape/submissions/htc_noGA_merge/merge3_iou0.7/' + 'results2/'
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)
    parse_pkldets(input_pkl, img_dir, out_dir, score_thr=0.01)


if __name__ == '__main__':
    main()



