pt=[410,234,450,235,451,190,411,54]
pt=[1092,38,1184,8,1196,0,1087,39]
import cv2
img=cv2.imread('./demo/img_470.jpg')
cv2.circle(img,(pt[0],pt[1]),1,(0,255,0),5)
cv2.circle(img,(pt[2],pt[3]),1,(255,255,255),5)
cv2.circle(img,(pt[4],pt[5]),1,(255,0,0),5)
cv2.circle(img,(pt[6],pt[7]),1,(0,0,255),5)
cv2.imshow('img',img)
cv2.waitKey(0)

from scipy.spatial import distance as dist
import numpy as np
import math
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl - tl)
    vector_1 = np.array(rightMost[0] - tl)
    vector_2 = np.array(rightMost[1] - tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
pt=np.array(pt)
pt_temp=np.reshape(pt,(4,2))
pt=order_points_quadrangle(pt_temp)
print(pt)