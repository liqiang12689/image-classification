import cv2 as cv
import numpy as np
import cv2

def get_avg(list5):
    if len(list5)==0:
        avg=0
    else:
        avg = sum(list5)/ len(list5)
    return avg

def light(img1,img2):
    x = img1.shape[0]
    y = img1.shape[1]
    for i in range(x-1):
        for j in range(y-1):
            b = int(img1[i, j + 1][0]) - int(img1[i, j][0])
            g = int(img1[i, j + 1][1]) - int(img1[i, j][1])
            r = int(img1[i, j + 1][2]) - int(img1[i, j][2])
            print(b,g,r)
            img2[i, j + 1][0] = img2[i, j][0] + b
            img2[i, j + 1][1] = img2[i, j][1] + g
            img2[i, j + 1][2] = img2[i, j][2] + r
    cv2.imwrite('img/26.jpg',img2)

img1= cv2.imread('img/23.jpg')
img2= cv2.imread('img/25.jpg')
light(img1,img2)
