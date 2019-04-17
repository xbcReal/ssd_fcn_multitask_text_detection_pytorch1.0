import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import math
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    poly_ = np.array(poly)
    assert poly_.shape == (4,2), 'poly shape should be 4,2'
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.
class icdar2015dataset(torch.utils.data.Dataset):
    class_names = ('__background__','text')

    def __init__(self,thresh,transform=None, target_transform=None, keep_difficult=True,keep_difficult_in_score_map=True,filter_small_quad=True):
        "Dataset for icdar2015 data."
        self.img_dir = '/home/binchengxiong/ocr_data/ICDAR2015/ch4_training_images/'
        self.gt_dir = '/home/binchengxiong/ocr_data/ICDAR2015/ch4_training_localization_transcription_gt_for_train/'
        self.transform = transform
        self.target_transform = target_transform
        self.img_file_names=os.listdir(self.img_dir)
        self.gt_file_names=[os.path.join(self.gt_dir+'gt_'+item) for item in self.img_file_names]
        self.img_file_names=[os.path.join(self.img_dir,item) for item in self.img_file_names]
        self.gt_file_names=[item.replace('.jpg','.txt') for item in self.gt_file_names]
        # for img,gt in zip(img_file_names,gt_file_names):
        #     print(img)
        #     print(gt)
        self.keep_difficult = keep_difficult
        self.keep_difficult_in_score_map = keep_difficult_in_score_map
        self.filter_small_quad = filter_small_quad#滤出太小的文本区域
        self.thresh = thresh
        self.read_times = 0
        self.ignore_small_quad_in_ssd = True
        self.ignore_small_quad_in_fcn = True

    def __getitem__(self, index):
        self.read_times += 1
        image_name=self.img_file_names[index]
        # print(image_name)
        image = self.read_image(image_name)
        gt_name=self.gt_file_names[index]
        #quad one element form:[x1, y1, x2, y2, x3, y3, x4, y4]
        quad, is_difficult = self.get_annotation(gt_name)
        quad_for_score_map = quad
        if not self.keep_difficult:
            is_easy=(1-is_difficult).astype(np.bool)
            quad_temp = quad[is_easy]
            difficult_quads = quad[is_difficult.astype(np.bool)]
            if(len(quad_temp) == 0):
                quad = quad
            else:
                quad = quad_temp
                image = np.float32(image)
                difficult_quads = np.reshape(difficult_quads, (-1, 4, 2))
                difficult_quads = np.int32(difficult_quads)
                import random
                color_b = random.randint(0,255)
                color_g = random.randint(0,255)
                color_r = random.randint(0,255)
                cv2.fillPoly(image, difficult_quads, (color_b, color_g, color_r))
        #else:
            # if self.filter_small_quad:
            #     is_easy = (1 - is_difficult).astype(np.bool)
            #     is_difficult = is_difficult.astype(np.bool)
            #     quad_easy = quad[is_easy]
            #     quad_difficult = quad[is_difficult]



        if not self.keep_difficult_in_score_map:
            is_easy=(1-is_difficult).astype(np.bool)
            quad_temp = quad_for_score_map[is_easy]
            if(len(quad_temp) == 0):
                quad_for_score_map = quad_for_score_map
            else:
                quad_for_score_map = quad_temp
        score_map = self.get_score_map(image,quad_for_score_map)
        # score_map = None
        if self.transform:
            image, quad ,score_map= self.transform(image,quad,score_map)
        # if True:
        #     img=np.swapaxes(image,0,1)
        #     img=np.swapaxes(img,1,2)
        #     img=np.uint8(img)
        #     img_prior=img[:,:,:]
        #     quad=quad*512
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     img2 = img.copy()
        #     for i in range(np.shape(quad)[0]):
        #         cv2.line(img2, (quad[i][0], quad[i][1]), (quad[i][2], quad[i][3]), (255,255, 255), thickness=2)
        #         cv2.line(img2, (quad[i][2], quad[i][3]), (quad[i][4], quad[i][5]), (255,255, 255), thickness=2)
        #         cv2.line(img2, (quad[i][4], quad[i][5]), (quad[i][6], quad[i][7]), (255,255, 255), thickness=2)
        #         cv2.line(img2, (quad[i][6], quad[i][7]), (quad[i][0], quad[i][1]), (255,255, 255), thickness=2)
        #     quad = quad / 512.0
        #     cv2.imshow('img_gt2',img2)
        # else:
        #     img_prior=None
        if self.keep_difficult and self.filter_small_quad:
            quad_copy = []
            thresh = self.thresh
            # print('trhesh:',thresh)
            # print('before fileter:',np.shape(quad))
            for i in range(np.shape(quad)[0]):
                # if (abs(quad[i][0] - quad[i][2]) < thresh) or (abs(quad[i][2] - quad[i][4]) < thresh) or (abs(quad[i][4] - quad[i][6]) < thresh) or (abs(quad[i][6] - quad[i][0]) < thresh):
                #     continue
                # elif (abs(quad[i][1] - quad[i][3]) < thresh) or (abs(quad[i][3] - quad[i][5]) < thresh) or (abs(quad[i][5] - quad[i][7]) < thresh) or (abs(quad[i][7] - quad[i][1]) < thresh):
                #     continue
                # else:
                #     quad_copy.append(quad[i])
                if math.sqrt(math.pow(quad[i][0] - quad[i][2],2)+math.pow(quad[i][1] - quad[i][3],2)) < thresh:
                    # print(math.pow(quad[i][0] - quad[i][2],2)+math.pow(quad[i][1] - quad[i][3],2))
                    # print('filter some small boxes')
                    continue
                elif math.sqrt(math.pow(quad[i][2] - quad[i][4],2)+math.pow(quad[i][3] - quad[i][5],2)) < thresh:
                    # print('filter some small boxes')
                    # print(math.pow(quad[i][2] - quad[i][4],2)+math.pow(quad[i][3] - quad[i][5],2))
                    continue
                elif math.sqrt(math.pow(quad[i][4] - quad[i][6],2)+math.pow(quad[i][5] - quad[i][7],2)) < thresh:
                    # print('filter some small boxes')
                    # print(math.pow(quad[i][4] - quad[i][6],2)+math.pow(quad[i][5] - quad[i][7],2))
                    continue
                elif math.sqrt(math.pow(quad[i][6] - quad[i][0],2)+math.pow(quad[i][7] - quad[i][1],2)) < thresh:
                    # print('filter some small boxes')
                    # print(math.pow(quad[i][6] - quad[i][0],2)+math.pow(quad[i][7] - quad[i][1], 2))
                    continue
                else:
                    quad_copy.append(quad[i])
            quad = np.array(quad_copy)
            # print('after fileter:',np.shape(quad))

        if score_map is None:
            pass
        else:
            score_map = score_map.numpy()
            score_map = cv2.resize(score_map,(128,128),interpolation=cv2.INTER_CUBIC)

        if False:
            img=np.swapaxes(image,0,1)
            img=np.swapaxes(img,1,2)
            img=np.uint8(img)
            img_prior=img[:,:,:]
            quad=quad*512
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img1 = img.copy()
            for i in range(np.shape(quad)[0]):
                cv2.line(img1, (int(quad[i][0]), int(quad[i][1])), (int(quad[i][2]), int(quad[i][3])), (255,255, 255), thickness=2)
                cv2.line(img1, (int(quad[i][2]), int(quad[i][3])), (int(quad[i][4]), int(quad[i][5])), (255,255, 255), thickness=2)
                cv2.line(img1, (int(quad[i][4]), int(quad[i][5])), (int(quad[i][6]), int(quad[i][7])), (255,255, 255), thickness=2)
                cv2.line(img1, (int(quad[i][6]), int(quad[i][7])), (int(quad[i][0]), int(quad[i][1])), (255,255, 255), thickness=2)
            quad = quad / 512.0
            cv2.imshow('img_gt1',img1)
            # cv2.imshow('scoremap',score_map)
            # cv2.waitKey()
        else:
            img_prior=None

        #type(quad):numpy.ndarray
        # np.savetxt('quad.txt',quad)

        if self.target_transform:
            quad,labels = self.target_transform(quad,img_prior)
        if score_map is None:
            return image,quad,labels,1
        else:
            return image, quad,labels,score_map

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_file_names)


    def get_annotation(self, gt_name):
        # print('gt_name:',self.read_times,gt_name)
        annotations=[]
        is_difficult=[]
        # print('gt_name',gt_name)
        with open(gt_name,encoding='UTF-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                #print('line:',line)
                #print(len(line.strip().split(',')))
                try:
                    x1, y1, x2, y2, x3, y3, x4, y4, transcription=line.strip().split(',')
                except ValueError:
                    print(gt_name)
                    exit()
                if '#' in transcription:
                    hard=True
                else:
                    hard=False
                annotation=[float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]

                is_difficult.append(hard)
                annotations.append(annotation)
        if annotations==[]:
            print('annotations==[]')
            #exit(0)
        if is_difficult.count('True') == len(annotations):
            print('there are all difficult instances')
            exit(0)
        return np.array(annotations,dtype=np.float32),np.array(is_difficult,dtype=np.uint8)

    def read_image(self, image_name):
        image = Image.open(image_name).convert("RGB")
        image = np.array(image)
        #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        return image
    def get_score_map(self,image,quad_for_score_map):
        h, w ,_= np.shape(image)
        score_map = np.zeros((h, w), dtype=np.float32)
        quad_for_score_map = np.int32(quad_for_score_map)
        quad_for_score_map = np.reshape(quad_for_score_map,(-1,4,2))
        cv2.fillPoly(score_map, quad_for_score_map, 1)
        show_score_map = False
        if show_score_map:
            cv2.imshow('image',image)
            cv2.imshow('score_map',score_map)
            cv2.waitKey()
        return score_map


