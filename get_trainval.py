import os
import random

# xml_path = 'annotations_aug\\'
# xml_files = os.listdir('annotations_aug')
 
trainval_path = '/home/caohaotian/WorkSpace/Data/org_trainval.txt'
train_file = '/home/caohaotian/WorkSpace/Data/org_train.txt'
test_file = '/home/caohaotian/WorkSpace/Data/org_test.txt'
# augment_file_all = 'train_aug_all.txt'

with open(test_file,'w') as test_writer:
    with open(train_file,'w') as train_writer:
        with open(trainval_path,'r') as f:
            for line in f:
                random_num = random.randint(0,10)
                if random_num <= 8 or int(line.split('_')[0])==2007:
                    train_writer.write(line)
                else:
                    test_writer.write(line)
"""
files = os.listdir('Annotations\\')
with open(test_file,'w') as test_writer:
    with open(train_file,'w') as train_writer:
        with open(trainval_path,'r') as f:
            for line in files:
                random_num = random.randint(0,10)
                if random_num <= 8 and int(line.split('_')[0])<=2012:
                    train_writer.write(line.split('.')[0]+'\n')
                elif int(line.split('_')[0])<=2012:
                    test_writer.write(line.split('.')[0]+'\n')
"""
"""
i = 0
with open(train_file) as f:
    with open(augment_file_all,'w') as aug:
        for line in f:
            num_str = line.strip('\n').split('_')
            num = int(int(num_str[0]) + 6)
            suffix = num_str[1]
            all_name = str(num)+'_'+suffix
            if os.path.exists('D:\\dataset\\VOCdevkit\\VOC2012\\Annotations\\'+all_name+'.xml'):
                aug.write(all_name+'\n')
            else:
                i = i + 1
                print(all_name)
print(i)
"""
