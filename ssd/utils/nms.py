import torch_extension
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
import numpy as np
_nms = torch_extension.nms


def boxes_nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(not support relative coordinates),
            shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    keep = _nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep
def iou_shapely(quad1,quad2):
    poly1 = Polygon(quad1.reshape(4, 2)).convex_hull
    poly2 = Polygon(quad2.reshape(4, 2)).convex_hull

    union_poly = np.concatenate((quad1.reshape(4, 2), quad2.reshape(4, 2)))  # 合并两个box坐标，变为8*2
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0
            else:
                iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou
def quad_boxes_nms(boxes,scores,nms_thresh,max_count=-1):

    # print('boxes.size:',boxes.size())
    # print('scores.size:',scores.size())
    boxes=boxes.cpu().numpy()
    scores=scores.cpu().numpy()
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        box_i=boxes[i]
        other_boxes=boxes[order[1:]]
        iou_result=[]
        for j in range(len(other_boxes)):
            iou = iou_shapely(box_i,other_boxes[j])
            iou_result.append(iou)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        iou_result=np.array(iou_result)
        inds = np.where(iou_result <= nms_thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return keep
