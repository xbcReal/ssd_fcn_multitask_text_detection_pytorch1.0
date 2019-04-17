import  numpy as np
import os
def get_annotation(gt_name):
    print('gt_name:', gt_name)
    annotations = []
    is_difficult = []
    with open(gt_name, encoding='UTF-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            print('line:',line)
            # print(len(line.strip().split(',')))
            x1, y1, x2, y2, x3, y3, x4, y4, transcription = line.strip().split(',')
            if '#' in transcription:
                hard = 1
            else:
                hard = 0
            annotation = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]

            is_difficult.append(hard)
            annotations.append(annotation)
        # exit(0)
    return np.array(annotations, dtype=np.float32), np.array(is_difficult, dtype=np.uint8)
gt_file_root='/home/binchengxiong/ocr_data/ICDAR2015/ch4_training_localization_transcription_gt_for_train/'
another_root='/home/binchengxiong/ocr_data/ICDAR2015/another_root'
file_names=os.listdir(gt_file_root)
#file_names = [os.path.join(gt_file_root,item) for item in file_names]
for item in file_names:
    f1 = open(os.path.join(gt_file_root,item),'r',encoding='utf-8-sig')
    contents = f1.read()
    print(contents)
    f2 = open(os.path.join(another_root,item),"w", encoding='utf-8-sig')
    #lines = f1.readlines()

    #u = [str(line) for line in lines]
    f2.write(contents)
    f1.close()
    f2.close()
