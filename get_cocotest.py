import os
import random

xml_dir = '/home/cao/workspace/PASCAL_VOC/Dataset/MSCOCO/VOCdevkit/VOC2012/Annotations'
txt = './temp/coco_test1000.txt'

names = os.listdir(xml_dir)
inds = random.sample(range(len(names)), 2000)

with open(txt,'w') as f:
    for i in inds:
        f.write(names[i].split('.')[0]+'\n')
    
