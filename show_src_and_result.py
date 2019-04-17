import cv2
for i in range(300,500):
    img1=cv2.imread('demo/result1/img_'+str(i+1)+'.jpg')
    img2=cv2.imread('demo/result/img_'+str(i+1)+'.jpg')
    cv2.imshow('img1',img1)
    cv2.imshow('img2',img2)
    cv2.waitKey()