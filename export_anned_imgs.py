import os
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np


color_map = {
    'aeroplane': (0, 245, 255),
    'bicycle': (0, 0, 255),
    'bird': (255, 255, 0),
    'boat': (255, 130, 71),
    'bottle': (255, 0, 255),
    'bus': (155, 48, 255),
    'car': (255, 99, 71),
    'cat': (147, 112, 219),
    'chair': (131, 111, 255),
    'cow': (0, 191, 255),
    'diningtable': (255, 0, 0),
    'dog': (255, 130, 71),
    'horse': (238, 122, 233),
    'motorbike': (255, 106, 106),
    'person': (0, 255, 0),
    'pottedplant': (0, 255, 255),
    'sheep': (255, 193, 37), 
    'sofa': (255, 165, 0),
    'train': (255, 20, 147),
    'tvmonitor': (255, 130, 71)
}  


def get_ann_info(img_prefix, img_id):
    # img_prefix = ''
    xml_path = osp.join(img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        label = name
        try:
            difficult = int(obj.find('difficult').text)
        except:
            difficult = False

        bnd_box = obj.find('bndbox')
        bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
               ]
        
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def export(intxt, outdir, img_prefix):
    img_names_list = []

    for line in open(intxt):
        img_names_list.append(line[:-1])

    k = 0
    for img_id in img_names_list:
        bboxes, labels = get_ann_info(img_prefix, img_id)
        if len(bboxes) >= 20:
            img = cv2.imread(osp.join(img_prefix, 'JPEGImages',
                            '{}.jpg'.format(img_id)))
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                label = labels[i]
                color = color_map[label]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                img[bbox[1]-12:bbox[1], bbox[0]:bbox[0]+int(len(label)*7)] = color
                cv2.putText(img, label, (bbox[0], bbox[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                cv2.imwrite(outdir+img_id+'.jpg', img)
        if k % 100 == 0:
            print(k)
        k += 1
 

if __name__ == '__main__':
    img_prefix = '/home/cao/workspace/PASCAL_VOC/Dataset/VOC_0712_All_noAug/VOC2012/'
    intxt = img_prefix + 'ImageSets/all_imgs.txt'
    outdir = '/media/cao/0EF30E3D0EF30E3D/DatasetBackup/VOC_0712_All_noAug_AnnoImgs/JPEGImages/'
    export(intxt, outdir, img_prefix)

 
         
