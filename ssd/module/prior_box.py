from itertools import product

import torch
import torch.nn as nn
from math import sqrt
import numpy as np
import random
import math
class PriorBox(nn.Module):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP
        # 用于旋转的prior box的参数
        self.rotate_theta_range = cfg.MODEL.PRIORS.ROTATE_THATA_RANGE
        self.divided_num = cfg.MODEL.PRIORS.DIVIDED_NUM
        self.rotate_thetas = list(np.linspace(3.14 / 180 * self.rotate_theta_range[0],3.14 / 180 * self.rotate_theta_range[1],self.divided_num))[1:-1]


    def forward(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """

        print('rotate_thetas:',self.rotate_thetas)
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            if len(self.aspect_ratios[k])==0:
                continue
            #全排列
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                #small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                #quad form:x1,y1,x2,y2,x3,y3,x4,y4
                priors.append([cx-w/2.0, cy-h/2.0,cx+w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0,cx-w/2.0,cy+h/2.0])
                for theta in self.rotate_thetas:
                    priors.append(self.rotate_prior_box(
                        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, cx - w / 2.0,
                         cy + h / 2.0], cx, cy, theta))

                #big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx-w/2.0, cy-h/2.0,cx+w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0,cx-w/2.0,cy+h/2.0])
                for theta in self.rotate_thetas:
                    priors.append(self.rotate_prior_box(
                        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, cx - w / 2.0,
                         cy + h / 2.0], cx, cy, theta))

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append(
                        [cx - w * ratio / 2.0, cy - h / ratio / 2.0, cx + w * ratio / 2.0, cy - h / ratio / 2.0,
                         cx + w * ratio / 2.0, cy + h / ratio / 2.0, cx - w * ratio / 2.0, cy + h / ratio / 2.0])
                    for theta in self.rotate_thetas:
                        priors.append(self.rotate_prior_box(
                            [cx - w * ratio / 2.0, cy - h / ratio / 2.0, cx + w * ratio / 2.0, cy - h / ratio / 2.0,
                             cx + w * ratio / 2.0, cy + h / ratio / 2.0, cx - w * ratio / 2.0, cy + h / ratio / 2.0], cx, cy, theta))
                    priors.append(
                        [cx - w / ratio / 2.0, cy - h * ratio / 2.0, cx + w / ratio / 2.0, cy - h * ratio / 2.0,
                         cx + w / ratio / 2.0, cy + h * ratio / 2.0, cx - w / ratio / 2.0, cy + h * ratio / 2.0])
                    for theta in self.rotate_thetas:
                        priors.append(self.rotate_prior_box(
                            [cx - w / ratio / 2.0, cy - h * ratio / 2.0, cx + w / ratio / 2.0, cy - h * ratio / 2.0,
                             cx + w / ratio / 2.0, cy + h * ratio / 2.0, cx - w / ratio / 2.0, cy + h * ratio / 2.0], cx, cy, theta))


        #priors = torch.Tensor(priors)
        priors=np.array(priors)
        print('prior.size:',np.shape(priors))
        if True:
            priors=np.clip(priors, 0, 1)
        return priors.astype(np.float32)
    def rotate_prior_box(self,quad,cx,cy,theta):
        # 计算公式参考https://jingyan.baidu.com/article/2c8c281dfbf3dd0009252a7b.html
        # x0 = (x - rx0) * cos(a) - (y - ry0) * sin(a) + rx0
        # y0 = (x - rx0) * sin(a) + (y - ry0) * cos(a) + ry0
        rotate_theta=theta
        # print('rotate_theta:',rotate_theta)
        xs=np.array(quad[::2],np.float32)
        ys=np.array(quad[1::2],np.float32)
        rotate_xs=(xs-cx)*math.cos(rotate_theta)-(ys-cy)*math.sin(rotate_theta)+cx
        rotate_ys=(xs-cx)*math.sin(rotate_theta)+(ys-cy)*math.cos(rotate_theta)+cy
        rotate_quad=[]
        for i in range(4):
            rotate_quad.append(rotate_xs[i])
            rotate_quad.append(rotate_ys[i])
        assert len(rotate_quad) == 8
        if theta > 45 * 3.14 / 180:
            tmp = rotate_quad[2:]
            tmp1 = rotate_quad[0:2]
            tmp.extend(tmp1)
            assert len(tmp) == 8
            return np.clip(tmp,0,1)
        return np.clip(rotate_quad,0,1)

