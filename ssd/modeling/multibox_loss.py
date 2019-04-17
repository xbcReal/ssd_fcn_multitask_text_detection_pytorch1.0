import torch.nn as nn
import torch.nn.functional as F
import torch
from ssd.utils import box_utils
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1,1).long()
        logpt = F.log_softmax(input,dim=2).view(-1,2)
        #print('logpt:',logpt)
        # print('logpt.size:',logpt.size())
        # print('target.size:',target.size())
        logpt = logpt.gather(1,target)
        # print('logpt:',logpt)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            # print('at.size:',at.size())
            at = torch.unsqueeze(at, 1)
            # print('at.count0.25',torch.sum(at==0.25))
            # print('at.count0.75',torch.sum(at==0.75))
            # print('logpt.size:',logpt.size())
            logpt = logpt * Variable(at)
            # print('logpt.size:',logpt.size())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.focalloss = FocalLoss()

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        # print('confidence.size1:', confidence.size())  # [bs,num_anchors,2]
        # print('labels.size:', labels.size())  # [bs,num_anchors]
        num_classes = 2
        #classification_loss = self.focalloss(confidence,labels)
        # classification_loss = F.cross_entropy(confidence.view(-1,num_classes),labels.view(-1).long(),reduce='mean')
        with torch.no_grad():
        #     # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            # num_pos = torch.sum(labels==1)
            # print('num_pos:',num_pos)
            # print('num_posx4:', num_pos*4)
        # print('self.neg_pos_ratio:',self.neg_pos_ratio)
        mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        # print('confidence.size:', confidence.size())  # [bs,num_anchors,2]

        classification_loss = F.cross_entropy(confidence.view(-1,num_classes), labels[mask].long(), reduction='sum')

        pos_mask = labels > 0
        # print('predicted_locations.size:',predicted_locations.size())
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 8)
        gt_locations = gt_locations[pos_mask, :].view(-1, 8)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations.float(), reduction='sum')
        #l2_loss = F.mse_loss(predicted_locations, gt_locations.float(),reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos