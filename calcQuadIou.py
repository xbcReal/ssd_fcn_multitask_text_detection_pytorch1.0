#这个文件的是用来测试两种计算两个任意四边形的IOU的速度及其精度度
#这两个方法分别是：
#1.shapely包提供的计算方法，原理不明
#2.萨瑟兰-Hodgman算法
import shapely

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
    iou = intersection_area / (PolygonArea(quad1) + PolygonArea(quad2) - intersection_area)
    return iou
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
import numpy as np
import time
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


quad1=[[15,10],[30,10],[30,20],[10,20]]
quad2=[[20,15],[40,15],[40,30],[15,30]]
start = time.time()
for i in range(25000*4):
    iou1=calc_iou_Hodgman(quad1,quad2)
end = time.time()
print('iou1:',iou1)
print('time consuming1:',end-start)
start = time.time()
for i in range(25000*4):
    iou2 = iou_shapely(np.array(quad1),np.array(quad2))
end = time.time()
print('iou2:',iou2)
print('time consuming2:',end-start)

