import os

img_dir = '/home/cao/workspace/PASCAL_VOC/Dataset/MSCOCO/VOCdevkit/VOC2012/Annotations'
testtxt = '/home/cao/workspace/PASCAL_VOC/Dataset/MSCOCO/VOCdevkit/VOC2012/ImageSets/coco_test1000.txt'
traintxt = '/home/cao/workspace/PASCAL_VOC/Dataset/MSCOCO/VOCdevkit/VOC2012/ImageSets/train_imgs.txt'

alls = os.listdir(img_dir)
print(len(alls))

tests = []
trains = []

for line in open(testtxt):
    tests.append(line[:-1])

print(len(tests))

for name in alls:
    name_prex = name.split('.')[0]
    if name_prex not in tests:
        trains.append(name_prex)

    
print(len(trains))
with open(traintxt,'w') as f:
    for name in trains:
        f.write(name+'\n')



