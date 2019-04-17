import cv2
import matplotlib.pyplot as plt
import os
# count = 1
# path = './mserimg/'
# imgpaths=[]
# for i in range(1,135):
#     tmp_path = os.path.join(path,'img_'+str(i) + '.jpg')
#     imgpaths.append(tmp_path)
#
# for imgname in imgpaths:
#     print(imgname)
#     img = cv2.imread(imgname)
#     mser = cv2.MSER_create(_min_area=300)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     regions, boxes = mser.detectRegions(gray)
#
#     for box in boxes:
#         count += 1
#         x, y, w, h = box
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.imwrite('./mser/result'+str(count)+'.jpg',img[y:y+h,x:x+w,:])
#     # plt.imshow(img, 'brg')
#     # plt.show()


img = cv2.imread('img_11.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = 255 - img
cv2.imwrite('img.jpg',img)
threshold = range(0,255,10)
save_path = './mser1/'
for i in threshold:
    ret,thresh_img=cv2.threshold(img,i,255,cv2.THRESH_BINARY)
    cv2.imwrite(save_path+str(i)+'.jpg',thresh_img)
