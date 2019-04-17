import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from ssd.modeling.multibox_loss import MultiBoxLoss
from ssd.module import L2Norm
from ssd.module.prior_box import PriorBox
from ssd.utils import box_utils
from torch.nn.functional import binary_cross_entropy



class SSD(nn.Module):
    def __init__(self, cfg,
                 vgg: nn.ModuleList,
                 extras: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList,
                 downsample_layers_index:list):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.vgg = vgg
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.l2_norm = L2Norm(512, scale=20)
        self.criterion = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.priors = None
        self.downsample_layers_index = downsample_layers_index
        # FCN part
        self.fcn_module = []
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.fcn_module.append(self.conv1)
        self.bn1 = nn.BatchNorm2d(512)
        self.fcn_module.append(self.bn1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 64, 3, padding=1)
        self.fcn_module.append(self.conv2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fcn_module.append(self.bn2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool1_conv2d = nn.Conv2d(1024,512,1)
        self.fcn_module.append(self.unpool1_conv2d)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2_conv2d = nn.Conv2d(512,256,1)
        self.fcn_module.append(self.unpool2_conv2d)
        self.fcn_module = nn.ModuleList(self.fcn_module)
        self.reset_parameters()

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.vgg.apply(weights_init)
        self.extras.apply(weights_init)
        self.classification_headers.apply(weights_init)
        self.regression_headers.apply(weights_init)
        self.fcn_module.apply(weights_init)

    def dice_coefficient(self,y_true_cls, y_pred_cls):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :return:
        '''
        eps = 1e-5
        # print('y_true_cls:',y_true_cls.size)
        # print('y_pred_cls:',y_pred_cls.size)
        intersection = torch.sum(y_true_cls * y_pred_cls)
        union = torch.sum(y_true_cls) + torch.sum(y_pred_cls) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def balanced_cross_entropy(self,y_true_cls, y_pred_cls):
        # y_true_cls_cp=y_true_cls.clone()
        # y_true_cls_cp=y_true_cls.size()
        batch_size, w, h = y_true_cls.size()
        all_loss = 0.0
        for i in range(batch_size):
            true_count = torch.sum(y_true_cls[i])
            all_count = w * h
            beta = 1 - true_count / all_count #beta通常大于0.9
            y_pred_cls = y_pred_cls.data.cpu().numpy()
            y_pred_cls = np.clip(y_pred_cls, 0.1, 0.9)
            y_pred_cls = torch.from_numpy(y_pred_cls).cuda()
            all_loss = all_loss + torch.sum(-beta * y_true_cls[i] * torch.log(y_pred_cls[i]) - (1 - beta) * (1 - y_true_cls[i]) * torch.log(1 - y_pred_cls[i]))
        return all_loss / (batch_size*w*h)

    def balanced_cross_entropy_1(self,target,input):
        batch_size, w, h = target.size()
        pos_index = (target > 0.5)
        neg_index = (target < 0.5)
        weight = torch.Tensor(input.size()).fill_(0)
        pos_num = pos_index.sum().item()
        neg_num = neg_index.sum().item()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight = weight.cuda()

        loss = binary_cross_entropy(input, target, weight, reduction='none')

        return torch.sum(loss) / (batch_size*w*h)

    def forward(self, x, targets, score_map=None):
        # print('self.downsample_layers_index',self.downsample_layers_index)
        downsample_feature_map=[]
        sources = []
        confidences = []
        locations = []
        for i in range(23):
            x = self.vgg[i](x)
            # print('x.size():',x.size())
            # if i == 3:
            #     import os
            #     import glob
            #     path = '/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/img/tmp/'
            #     for infile in glob.glob(os.path.join(path, '*.jpg')):
            #         os.remove(infile)
            #     sizez = x.size()
            #     print('x.size:',x.size())
            #     for i in range(sizez[1]):
            #         tmp = x[0][i].cpu().numpy()
            #         max = tmp.max()
            #         min = tmp.min()
            #         print('max:',max)
            #         print('min:',min)
            #         featuremap = (tmp - min) / (max - min) * 255
            #
            #         featuremap = featuremap.astype(np.uint8)
            #         featuremap = cv2.applyColorMap(featuremap, cv2.COLORMAP_JET)
            #         cv2.imwrite(
            #             '/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/img/tmp/' + str(i) + '.jpg',
            #             featuremap)
            if i in self.downsample_layers_index:
                downsample_feature_map.append(x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        sources.append(s)
        # apply vgg up to fc7
        # print('len(vgg):',len(self.vgg))
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
            # print('x.size():',x.size())
            if i in self.downsample_layers_index:
                downsample_feature_map.append(x)
        sources.append(x)   #Conv_7

        # FCN part
        # for i in downsample_feature_map:
        #     print('i.size:',i.size())
        h = downsample_feature_map[2]  # bs 2048 w/32 h/32,f[3]是最后的输出层
        g = self.unpool1(h) # bs 2048 w/16 h/16
        g = self.unpool1_conv2d(g)
        # print('downsample_feature_map[2].size():',downsample_feature_map[2].size())
        c = self.conv1(g.add_(downsample_feature_map[1]))
        c = self.bn1(c)
        c = self.relu1(c)

        g = self.unpool2(c)  # bs 128 w/8 h/8
        g = self.unpool2_conv2d(g)
        c = self.conv2(g.add_(downsample_feature_map[0]))
        c = self.bn2(c)
        c = self.relu2(c)
        F_score = self.conv3(c)  # bs 1 w/4 h/4
        F_score = self.sigmoid(F_score)
        F_score = torch.squeeze(F_score)
        # print('F_score.size()',F_score.size())
        # print('score_map.size()',score_map.size())

        # for i in downsample_feature_map:
        #     print('i.size():',i.size())

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # print('x.size():',x.size())
            if k % 2 == 1:
                sources.append(x) #Conv_8_2,Conv_9_2,Conv_10_2,Conv_11_2

        for (x, l, c) in zip(sources, self.regression_headers, self.classification_headers):
            #原始的feature map的维度是NCHW,permute之后是NHWC
            a = l(x).permute(0, 2, 3, 1).contiguous()
            # print('a.size:',a.size())
            locations.append(a)
            b = c(x).permute(0, 2, 3, 1).contiguous()
            # print('b.size:',b.size())
            confidences.append(b)

        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)
        #print('locations.size()',locations.size())
        # print('self.num_classes:',self.num_classes)
        confidences = confidences.view(confidences.size(0), -1, self.num_classes)
        # print('confidence.size()',confidences.size())        #[batch_size,24564,2]
        locations = locations.view(locations.size(0), -1, 8)
        #print('locations.size()',locations.size())           #[batch_size,24564,8]

        if not self.training:
            # print('test')
            # when evaluating, decode predictions
            if self.priors is None:
                self.priors = PriorBox(self.cfg)()
            confidences = F.softmax(confidences, dim=2)
            quad = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
            )
            score_map = F_score.cpu()
            return confidences, quad,score_map
        else:
            # when training, compute losses
            gt_boxes, gt_labels = targets
            # print('locations:',locations)
            #给了事先匹配好的default box的位置和类别作为真值，回归预测的confidences和locations
            regression_loss, classification_loss = self.criterion(confidences, locations, gt_labels, gt_boxes)
            seg_loss = self.dice_coefficient(score_map,F_score)
            #seg_loss = self.balanced_cross_entropy(score_map,F_score)
            #seg_loss = self.balanced_cross_entropy_1(score_map,F_score)
            loss_dict = dict(
                regression_loss=regression_loss,
                classification_loss=classification_loss,
                fcn_loss=seg_loss
            )
            return loss_dict

    def init_from_base_net(self, model):
        vgg_weights = torch.load(model, map_location=lambda storage, loc: storage)
        self.vgg.load_state_dict(vgg_weights, strict=True)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

import cv2
class MatchPrior(object):
    def __init__(self, quad_form_priors, center_variance, size_variance, iou_threshold,distance_threshold):
        self.quad_form_priors = quad_form_priors
        #self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.distance_threshold=distance_threshold
        print('distance_threshold:',distance_threshold)

    def __call__(self, quad_gt,img_prior=None):

        #将np array转为torch tensor
        # if type(quad) is np.ndarray:
        #     quad = torch.from_numpy(quad)
        priors_boxes,prior_boxed_labels = box_utils.assign_priors(quad_gt,self.quad_form_priors, self.iou_threshold,self.distance_threshold)

        # print('prior_boxed_labels:',prior_boxed_labels)
        true_default_box_index=np.where(prior_boxed_labels==1)
        true_default_box=self.quad_form_priors[true_default_box_index] * 512
        # print('self.quad_form_prior.shape:',np.shape(self.quad_form_priors))
        tmp = self.quad_form_priors[382200:382270,:] * 512
        # print('tmp.shape:',np.shape(tmp))
        # print('true_default_box.shape:',np.shape(true_default_box))
        if img_prior is not None:
            for i in range(np.shape(true_default_box)[0]):
                print(true_default_box[i])
                # print(quad[i])
                cv2.line(img_prior, (true_default_box[i][0], true_default_box[i][1]), (true_default_box[i][2], true_default_box[i][3]), (255, 0, 0), thickness=2)
                cv2.line(img_prior, (true_default_box[i][2], true_default_box[i][3]), (true_default_box[i][4], true_default_box[i][5]), (255, 0, 0), thickness=2)
                cv2.line(img_prior, (true_default_box[i][4], true_default_box[i][5]), (true_default_box[i][6], true_default_box[i][7]), (255, 0, 0), thickness=2)
                cv2.line(img_prior, (true_default_box[i][6], true_default_box[i][7]), (true_default_box[i][0], true_default_box[i][1]), (255, 0, 0), thickness=2)
            for i in range(np.shape(tmp)[0]):
                # print('tmp[i]',tmp[i])
                # print('i:',i)
                cv2.line(img_prior, (tmp[i][0], tmp[i][1]), (tmp[i][2], tmp[i][3]), (0, 0, 255), thickness=2)
                cv2.line(img_prior, (tmp[i][2], tmp[i][3]), (tmp[i][4], tmp[i][5]), (0, 0, 255), thickness=2)
                cv2.line(img_prior, (tmp[i][4], tmp[i][5]), (tmp[i][6], tmp[i][7]), (0, 0, 255), thickness=2)
                cv2.line(img_prior, (tmp[i][6], tmp[i][7]), (tmp[i][0], tmp[i][1]), (0, 0, 255), thickness=2)
            cv2.imshow('img_prior', img_prior)
            cv2.imwrite('test.jpg',img_prior)
            cv2.waitKey()
        # print('priors_boxes.shape',np.shape(priors_boxes))
        locations = box_utils.convert_boxes_to_locations(priors_boxes, self.quad_form_priors, self.center_variance, self.size_variance)
        #locations.size():[24564,8]
        #prior_boxed_labels.size():[24564,]
        return locations,prior_boxed_labels
