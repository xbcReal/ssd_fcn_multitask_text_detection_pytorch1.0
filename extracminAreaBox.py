# import cv2
#
# img = cv2.imread('/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/demo/result2/img_160_score_map.jpg')
# print(img)

import cv2
import numpy as np
import time
img = cv2.imread('/home/binchengxiong/3_score_map.jpeg')
img1 = cv2.imread('/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/demo/result2/img_160_score_map1.jpg')
t1 = time.clock()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

w,h = np.shape(gray)
for contour in contours:
    print(1)
    # 获取最小包围矩形
    rect = cv2.minAreaRect(contour)

    # 中心坐标
    x, y = rect[0]
    # cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)

    # 长宽,总有 width>=height
    width, height = rect[1]
    if width < 10 or height < 10:
        continue

    # 角度:[-90,0)
    angle = rect[2]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box[:,0] = np.clip(box[:,0],0,h)
    box[:,1] = np.clip(box[:,1],0,w)
    cv2.line(img,(box[0][0],box[0][1]),(box[1][0],box[1][1]),(0,0,255),thickness=4)
    cv2.line(img,(box[1][0],box[1][1]),(box[2][0],box[2][1]),(0,0,255),thickness=4)
    cv2.line(img,(box[2][0],box[2][1]),(box[3][0],box[3][1]),(0,0,255),thickness=4)
    cv2.line(img,(box[3][0],box[3][1]),(box[0][0],box[0][1]),(0,0,255),thickness=4)
t2 = time.clock()
print(t2-t1)
#
cv2.imshow("contour", img)
cv2.imshow("bina", binary)

cv2.waitKey()

