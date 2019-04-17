
'''Load image/labels/boxes from an annotation file.
The list file is like:
    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import cv2

class mlt2017dataset(data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        '''
        Args:
          root: (str) DB root ditectory.
          dataset: (str) Dataset name(dir).
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
          multi_scale: (bool) use multi-scale training or not.
        '''
        self.root_dir = '/home/binchengxiong/ocr_data/MLT2017/'
        self.transform = transform
        self.target_transform = target_transform
        self.fnames = []
        self.quades = []
        self.get_mlt()
    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) dataset index.
        Returns:
          image: (tensor) image array.
          boxes: (tensor) boxes array.
          labels: (tensor) labels array.
        '''
        # Load image, boxes and labels.
        fname = self.fnames[idx]
        image = Image.open(os.path.join(self.root_dir, fname)).convert("RGB")
        image = np.array(image)
        quad = self.quades[idx].copy()
        score_map = None
        #score_map = self.get_score_map(image, quad)
        if self.transform:
            image, quad,score_map= self.transform(image,quad,score_map)
        # score_map = score_map.numpy()
        # score_map = cv2.resize(score_map, (256, 256), interpolation=cv2.INTER_CUBIC)
        if False:
            img=np.swapaxes(image,0,1)
            img=np.swapaxes(img,1,2)
            img=np.uint8(img)
            quad=quad*512
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img1 = img.copy()
            for i in range(np.shape(quad)[0]):
                cv2.line(img1, (int(quad[i][0]), int(quad[i][1])), (int(quad[i][2]), int(quad[i][3])), (255,255, 255), thickness=2)
                cv2.line(img1, (int(quad[i][2]), int(quad[i][3])), (int(quad[i][4]), int(quad[i][5])), (255,0, 0), thickness=2)
                cv2.line(img1, (int(quad[i][4]), int(quad[i][5])), (int(quad[i][6]), int(quad[i][7])), (0,0, 255), thickness=2)
                cv2.line(img1, (int(quad[i][6]), int(quad[i][7])), (int(quad[i][0]), int(quad[i][1])), (0,0, 0), thickness=2)
            quad = quad / 512.0
            cv2.imshow('img_gt1',img1)
            cv2.waitKey()
        else:
            img_prior=None
        if self.target_transform:
            quad,labels= self.target_transform(quad,img_prior)
        if score_map is None:
            return image,quad,labels,1
        else:
            return image, quad,labels,score_map
    def __len__(self):
        return self.num_samples

    def get_mlt(self):
        self.num_samples = 0
        train_dir = os.path.join(self.root_dir, 'train/')
        val_dir = os.path.join(self.root_dir, 'val/')
        dirs = [train_dir,val_dir]
        for dir in dirs:
            dataset_list = os.listdir(dir + "images")
            dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]
            for i in dataset_list:
                img_file = dataset_list + "%s.jpg" % (i)
                label_file = open(dir + 'txtes' + "gt_%s.txt" % (i))
                label_file = label_file.readlines()
                _quad = []
                for label in label_file:
                    _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, lang, txt = label.split(",")[:10]
                    if "###" in txt:
                        continue
                    # try:
                    #     _x0 = int(_x0)
                    # except:
                    #     _x0 = int(_x0[1:])

                    _x0,_y0, _x1, _y1, _x2, _y2, _x3, _y3 = [int(p) for p in [_x0,_y0, _x1, _y1, _x2, _y2, _x3, _y3]]

                    _quad.append([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3])

                if len(_quad) is not 0:
                    self.num_samples += 1
                    self.fnames.append(img_file)
                    self.quades.append(np.array(_quad, dtype=np.float32))
                else:
                    continue
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




