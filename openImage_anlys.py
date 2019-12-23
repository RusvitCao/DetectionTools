import os
import csv
import json
 
anno_dir = '/home/caohaotian/WorkSpace/Data/org_trainval.csv'
save_dir = ''
label_dic = {}

id_row = 0
with open(anno_path, 'r') as file:
    file_csv = csv.reader(file)
    for row in file_csv:
        print(id_row)
        if id_row != 0:
            print(row)
            # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
            if row[2] not in label_dic: # LabelName
                label_dic[row[2]] = {'num':1, 'Imgs':[row[0]]}
            else:
                label_dic[row[2]]['num'] += 1
                if row[0] not in label_dic[row[2]]['Imgs']:
                    label_dic[row[2]]['Imgs'].append(row[0])
        id_row += 1

with open(save_dir, 'w') as f:
    json.dump(label_dic, f)


