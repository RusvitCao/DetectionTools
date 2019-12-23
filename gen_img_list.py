import os
import random

img_dir = '/home/cao/workspace/PASCAL_VOC/Dataset/VOC_0712_All_noAug/VOC2012/JPEGImages'
txt = '/home/cao/workspace/PASCAL_VOC/Dataset/VOC_0712_All_noAug/VOC2012/ImageSets/all_imgs.txt'
testtxt = '/home/cao/workspace/PASCAL_VOC/Dataset/VOC_0712_All_noAug/VOC2012/ImageSets/test_imgs.txt'
traintxt = '/home/cao/workspace/PASCAL_VOC/Dataset/VOC_0712_All_noAug/VOC2012/ImageSets/train_imgs.txt'

names = os.listdir(img_dir)
test_names = []
train_names = []
# years = ['2008_', '2009_', '2010_', '2011_']


print('all:', len(names))
with open(txt,'w') as f:
    for name in names:
        f.write(name.split('.')[0]+'\n')
        # exist = False
        if int(name[:4]) < 2012 and int(name[:4]) > 2007:
            test_names.append(name.split('.')[0])
        else:
            train_names.append(name.split('.')[0])

print('train:', len(train_names))
print('2008~2011:', len(test_names))

inds = random.sample(range(len(test_names)), 1000)
with open(testtxt,'w') as f:
    for i in range(len(test_names)):
        if i in inds:
            f.write(test_names[i]+'\n')
        else:
            train_names.append(test_names[i])


print('test:', len(inds))
print('train:', len(train_names))


with open(traintxt,'w') as f:
    for name in train_names:
        f.write(name+'\n')
