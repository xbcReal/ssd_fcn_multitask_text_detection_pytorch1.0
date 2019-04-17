import torch
import math
import numpy as np

def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    # if priors.dim() + 1 == locations.dim():
    #     priors = priors.unsqueeze(0)
    # return torch.cat([
    #     locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
    #     torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    # ], dim=locations.dim() - 1)
    #print('locations:',locations)
    # print('priors.size():',priors.size)
    return locations*center_variance+torch.from_numpy(priors).cuda()


def convert_boxes_to_locations(quad_form_boxes, quad_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    # if center_form_priors.dim() + 1 == center_form_boxes.dim():
    #     center_form_priors = center_form_priors.unsqueeze(0)
    # return torch.cat([
    #     (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
    #     torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    # ], dim=center_form_boxes.dim() - 1)
    return (quad_form_boxes-quad_form_priors) / center_variance


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
from itertools import product
import time

#萨瑟兰-Hodgman算法
def clip(subjectPolygon, clipPolygon):
    def inside(p):
     return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
     dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
     dp = [ s[0] - e[0], s[1] - e[1] ]
     n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
     n2 = s[0] * e[1] - s[1] * e[0]
     n3 = 1.0/(dc[0] * dp[1] - dc[1] * dp[0])
     return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
     cp2 = clipVertex
     inputList = outputList
     outputList = []
     if inputList==[]:
         return [[0,0]]*4
     s = inputList[-1]

     for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
          if not inside(s):
           outputList.append(computeIntersection())
          outputList.append(e)
         elif inside(s):
          outputList.append(computeIntersection())
         s = e
     cp1 = cp2
    return(outputList)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
     j = (i + 1) % n
     area += corners[i][0] * corners[j][1]
     area -= corners[j][0] * corners[i][1]
    area = abs(area)/2.0
    return area
def calc_iou_Hodgman(quad1,quad2):
    intersection = clip(quad1, quad2)
    if intersection == 0:
        return 0
    intersection_area = PolygonArea(intersection)
    print('intersection_area:',intersection_area)
    print('PolygonArea(quad1):',PolygonArea(quad1))
    print('PolygonArea(quad2):',PolygonArea(quad2))
    print('PolygonArea(quad1) + PolygonArea(quad2):',PolygonArea(quad1) + PolygonArea(quad2))

    union_area=(PolygonArea(quad1) + PolygonArea(quad2) - intersection_area)
    print('union_area:',union_area)
    iou = intersection_area / union_area
    return iou
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (1,N,8): ground truth boxes.
        boxes1 (N,1,8): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    start = time.time()
    # print('boxes0.shape:',np.shape(boxes0))
    # print('boxes1.shape:',np.shape(boxes1))
    boxes0=np.reshape(boxes0,(-1,4,2))
    boxes1=np.reshape(boxes1,(-1,4,2))
    iou_result=np.zeros(shape=(np.shape(boxes1)[0],np.shape(boxes0)[0]),dtype=np.float32)
    for i, j in product(range(np.shape(boxes1)[0]),range(np.shape(boxes0)[0])):
        quad1=boxes0[j]
        quad2=boxes1[i]
        quad1=np.reshape(np.array(quad1),(4,2))
        quad2=np.reshape(np.array(quad2),(4,2))
        # iou=calc_iou_Hodgman(quad1,quad2)
        # if iou > 1 or iou < 0:
        #     print('iou:',iou)
        # assert iou <= 1 and iou >=0
        # iou_result[i][j] = iou
        poly1 = Polygon(quad1.reshape(4,2)).convex_hull
        poly2 = Polygon(quad2.reshape(4,2)).convex_hull

        union_poly = np.concatenate((quad1.reshape(4,2),quad2.reshape(4,2)))  # 合并两个box坐标，变为8*2
        if not poly1.intersects(poly2):  # 如果两四边形不相交
            iou = 0
        else:
            try:
                inter_area = poly1.intersection(poly2).area  # 相交面积
                #print(inter_area)
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0:
                    iou = 0
                else:
                    iou = float(inter_area) / union_area
                    iou_result[i][j] = iou
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured, iou set to 0')
                iou = 0
        assert iou <= 1 and iou >= 0
    end = time.time()
    #print('time consuming:',end-start)
    return iou_result


def distance_sum(quad_gt,quad_from_priors):
    ret = []
    # print('quad_gt.size:', np.shape(quad_gt))
    quad_gt=np.reshape(np.array(quad_gt),(-1,4,2))
    quad_from_priors=np.reshape(np.array(quad_from_priors),(-1,4,2))
    for i in range(np.shape(quad_gt)[0]):
        # ret_temp=b-a[i,:].sum(axis=1,keepdims=True)
        ret_temp = np.sum(np.sqrt(np.sum(np.power(quad_from_priors - quad_gt[i, ...],2), axis=2, keepdims=False)),axis=1,keepdims=True)
        #print('ret_temp.shape:',np.shape(ret_temp))
        ret.append(ret_temp)
    # print('ret.size:',len(ret))
    ret = np.concatenate(ret, axis=1)
    #print('ret.shape:', np.shape(ret))
    # print('quad_gt.shape:',np.shape(quad_gt))
    # print('quad_from_priors.shape:',np.shape(quad_from_priors))
    # print('ret.shape:',np.shape(ret))
    return ret
    # overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    # overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])
    #
    # overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    # area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    # area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    # return overlap_area / (area0 + area1 - overlap_area + eps)

def get_pos_distance_array(pos_distance_threshold):
    #根据不同尺度的default box自适应决定default box和gt距离的阈值
    # print('distance_threshold:',distance_threshold)
    # scale = [0.039,0.098,0.156,0.215,0.273,0.332,0.391]
    # diff_from_ratio = [1.656,1.588,1.491,1.403,1.323,1.261,1.203,1.068]#this if for different aspect ratio settings
    # diff_from_ratio = [1.656,1.656,1.656,1.656,1.656,1.656,1.656,1.656]
    # pos_distance_array = []
    # pos_distance_array += 64 * 64 * list(np.array([18 * [scale[0] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 32 * 32 * list(np.array([18 * [scale[1] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 16 * 16 * list(np.array([18 * [scale[2] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 8 * 8 * list(np.array([18 * [scale[3] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 4 * 4 * list(np.array([18 * [scale[4] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 2 * 2 * list(np.array([18 * [scale[5] * item] for item in diff_from_ratio]).reshape(-1))
    # pos_distance_array += 1 * 1 * list(np.array([18 * [scale[5] * item] for item in diff_from_ratio]).reshape(-1))
    # print('len(pos_distance_array):',len(pos_distance_array))
    # print('pos_distance_threshold:',pos_distance_threshold)
    n = 144
    pos_distance_array = []
    pos_distance_array+=64*64*n*[pos_distance_threshold[0]]#0~32768
    pos_distance_array+=32*32*n*[pos_distance_threshold[1]]#32768~40960
    pos_distance_array+=16*16*n*[pos_distance_threshold[2]]#40960~43008
    pos_distance_array+=8*8*n*[pos_distance_threshold[3]]#43008~43520
    pos_distance_array+=4*4*n*[pos_distance_threshold[4]]#43520~43648
    pos_distance_array+=2*2*n*[pos_distance_threshold[5]]#43648~43680
    pos_distance_array+=1*1*n*[pos_distance_threshold[6]]#43680~43688
    # print('distance_array.size:',np.shape(distance_array))
    # print('len:distance_array:',len(pos_distance_array))
    return np.array(pos_distance_array)
def get_ignore_distance_array(ignore_distance_threshold):
    #根据不同尺度的default box自适应决定default box和gt距离的阈值
    # print('distance_threshold:',distance_threshold)
    ignore_distance_array = []
    n = 126
    ignore_distance_array+=64*64*n*[ignore_distance_threshold[0]]#0~32768
    ignore_distance_array+=32*32*n*[ignore_distance_threshold[1]]#32768~40960
    ignore_distance_array+=16*16*n*[ignore_distance_threshold[2]]#40960~43008
    ignore_distance_array+=8*8*n*[ignore_distance_threshold[3]]#43008~43520
    ignore_distance_array+=4*4*n*[ignore_distance_threshold[4]]#43520~43648
    ignore_distance_array+=2*2*n*[ignore_distance_threshold[5]]#43648~43680
    ignore_distance_array+=1*1*n*[ignore_distance_threshold[6]]#43680~43688
    # print('distance_array.size:',np.shape(distance_array))
    return np.array(ignore_distance_array)

def assign_priors(quad_gt, quad_form_priors,iou_threshold,pos_distance_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    #ious = iou_of(quad_gt, quad_form_priors)
    #ious = iou_of(quad_gt, quad_form_priors)
    distance = distance_sum(quad_gt,quad_form_priors)


    # size: num_priors
    # 表示每一个prior对应distance最小的target的distance值
    best_target_per_prior=np.min(distance,axis=1)
    # 表示每一个prior对应distance最小的target的target的index值
    best_target_per_prior_index=np.argmin(distance,axis=1)
    #print(np.shape(best_target_per_prior))
    #print(np.shape(best_target_per_prior_index))
    # size: num_targets
    # 表示每一个target对应distance最小的prior的distance值
    best_prior_per_target=np.min(distance,axis=0)
    # 表示每一个target对应distance最小的prior的index
    best_prior_per_target_index=np.argmin(distance,axis=0)
    # 将每一个target对应的最大的prior赋值给这个prior对应最大的target
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior[best_prior_per_target_index]=2
    # size: num_priors
    gt_labels=np.ones(shape=np.shape(quad_gt)[0])
    labels = gt_labels[best_target_per_prior_index]
    # print('distance_threshold:',distance_threshold)
    pos_distance_array=get_pos_distance_array(pos_distance_threshold)
    ignore_distance_array=pos_distance_array * 1.995#1.995是根据曼哈顿距离度量中iou=0.3算出来的一个倍数关系

    labels[best_target_per_prior > pos_distance_array] = 0  # the backgournd id
    # print('shape:',np.shape(best_target_per_prior > pos_distance_array))
    #ignore_mask = np.multiply(best_target_per_prior > pos_distance_array ,best_target_per_prior < ignore_distance_array)
    # print('ignore_mask.size1:',ignore_mask.sum())
    #labels[ignore_mask] = -1
    quad = quad_gt[best_target_per_prior_index]
    # np.savetxt("/home/binchengxiong/boxes.txt", quad)
    # np.savetxt("/home/binchengxiong/labels.txt", labels)
    return quad,labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels == 1
    #ignore_mask = labels == -1
    # print('ignore_mask.size',ignore_mask.size())
    # print('ignore_mask2.size:',ignore_mask.sum())
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    # print('num_pos:',num_pos)
    num_neg = num_pos * neg_pos_ratio
    # print('pos_mask.size()[1]:',pos_mask.size()[1])
    # print('total train sample num:',num_pos * (neg_pos_ratio + 1))
    #把正样本对应的loss设为负无穷大，这样对loss进行降序排序的时候正样本的loss就会处于最后面
    # print('loss.size',loss.size())
    loss[pos_mask] = -math.inf
    #loss[ignore_mask] = -math.inf

    try:
        ordered_loss, indexes = loss.sort(dim=1, descending=True)
        # print('ordered_loss:',ordered_loss)
        # print('loss.size:',loss.size())
    except RuntimeError:
        print('loss.size()',loss.size())
        print('loss:',loss)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

#顶点形式的default box表示形式
def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
