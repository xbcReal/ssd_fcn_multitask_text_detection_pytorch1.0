import cv2
import numpy as np
image = np.ones((720,1280, 3), np.uint8) * 255
triangle = np.array([[1131,122,1161,125,1165,107,1119,123]])
triangle = np.reshape(triangle,(-1,4,2))
print(np.shape(triangle))
cv2.circle(image, (triangle[0][0][0], triangle[0][0][1]), 5, (0, 255, 0), 5)
cv2.circle(image, (triangle[0][1][0], triangle[0][1][1]), 5, (0, 0, 0), 5)
cv2.circle(image, (triangle[0][2][0], triangle[0][2][1]), 5, (255, 0, 0), 5)
cv2.circle(image, (triangle[0][3][0], triangle[0][3][1]), 5, (0, 0, 255), 5)

# triangle=np.int32(triangle)
# # print(np.shape(triangle))
# # print(triangle)
# img=cv2.fillPoly(img, triangle, (255, 255, 255))
#
cv2.imshow('img',image)
cv2.waitKey()
cv2.imwrite('black.jpg',image)