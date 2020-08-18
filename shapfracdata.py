import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from math import atan2,pi
import cv2

import itertools
import os
def unit_circle(d,r,mask,num=2):
    '''
    生成圆形
    d:图像边长
    r:蒙色直径
    '''
    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    mat1 = np.zeros((d, d))
    # mask = np.random.normal(size=(d,d), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
    rx , ry = int(d/2), int(d/2)
    for row in range(d):
        for col in range(d):
            dist = distance(rx, ry, row, col)
            # print((row,col),dist)
            if abs(dist) < r:
                mat1[row, col] = 1
            #黑白相间条纹
            # if row%20>16:
            #     # print(row,row/20)
            #     mat2[row, col] = 1
    #渐变黑白高斯纹理
    mat2 = unit_gauss(d,num)
    finalmat = np.multiply(mat1,mask)
    finalmat = np.multiply(mat2,finalmat)
    # plt.imsave('testcircle.png',finalmat)
    # Image.fromarray((finalmat * 255).astype('uint8'), mode='L').convert('RGB').save('testcircle.png')
    return finalmat

def unit_square(d,r,mask,num=2):
    '''
    生成正方形
    d:图像边长
    r:蒙色直径
    '''
    def distance(x1, y1, x2, y2):
        return abs(x1 - x2)
    mat1 = np.zeros((d, d))
    # mask = np.random.normal(size=(d,d), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
    rx , ry = int(d/2), int(d/2)
    for row in range(d):
        for col in range(d):
            # dist = distance(rx, ry, row, col)
            # print((row,col),dist)
            if abs(row-rx) < r and abs(col-ry) < r:
                mat1[row, col] = 1
            # if row%20>16:
            #     # print(row,row/20)
            #     mat2[row, col] = 1
    # 渐变黑白高斯纹理
    mat2 = unit_gauss(d,num)
    finalmat = np.multiply(mat1,mask)
    finalmat = np.multiply(mat2,finalmat)
    # plt.imsave('testsquare.png',finalmat)
    # Image.fromarray((finalmat * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare.png')
    return finalmat
def unit_triangle(d,r,mask,num=2):
    '''
    生成等边三角形
    d:图像边长
    r:蒙色直径
    '''
    mat1 = np.zeros((d, d))
    # mask = np.random.normal(size=(d,d), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
    for i in range(d):
        for j in range(d):
            if atan2(j, i) < pi / 3 and atan2(0 - j, d - i) > -pi / 3 and j>(d*80)//(5*r):
                mat1[i, j] = 1

    # 渐变黑白高斯纹理
    mat1 = unit_rot(mat1)
    # #移动
    # for i in range(d,0):
    #     if i < d-d//5:
    #         mat1[i,:] = mat1[i-d//5,:]
    mat2 = unit_gauss(d,num)
    finalmat = np.multiply(mat1,mask)
    finalmat = np.multiply(mat2,finalmat)
    # Image.fromarray((finalmat * 255).astype('uint8'), mode='L').convert('RGB').save('testangle.png')
    return finalmat

def unit_rot(data):
    # newdata45 = pcolormesh_45deg(data)
    newadta90 = np.rot90(data)
    # Image.fromarray((newdata45 * 255).astype('uint8'), mode='L').convert('RGB').save('testtor45.png')

    # Image.fromarray((newadta90 * 255).astype('uint8'), mode='L').convert('RGB').save('testtor90.png')
    return newadta90

def unit_gauss(N,num):
    '''
    N:矩阵边长
    num：条纹数量
    '''
    x = np.linspace(-5*num, 5*num, N)
    y = np.linspace(-5*num, 5*num, N)
    X,Y = np.meshgrid(x,y) #生成一个矩阵，并填充数据.
    Z = (np.sin(0.5 * np.pi * Y)+1)/2
    return Z


def Generateimage(numslice,classfic):
    dcm = np.zeros((numslice,512,512))
    dis = int(400/numslice)
    r = 0.
    i = 1
    for dic in range(numslice):
        r += dis
        if r>200:
            disr = r
            disr = disr-dis*(2*i-1)
            dcm[dic,:,:] = unit_circle(512,disr,classfic)
            i += 1
        else:
            disr = r
            dcm[dic,:,:] = unit_circle(512,disr,classfic)
    dcm = dcm.reshape(16,1,512,512)
    dcm = torch.from_numpy(dcm)
    img = torchvision.utils.make_grid(dcm,nrow=4,normalize=True)
    img = img.numpy().transpose((1,2,0))
    # plt.imsave('test.png',img)
    Image.fromarray((img * 255).astype('uint8'), mode='L').convert('RGB').save('test.png')
    print('max:',torch.max(dcm))
    # plt.imsave('test.png',finalmat)
    return dcm
def Generations(number):
    # classnum = int(number/5)
    y = []
    x = []
    for index in range(number):
        classfic = index%5 +1
        dcm = Generateimage(16,classfic)  #==>(16, 1, 512, 512)
        dcm = dcm.reshape(16,512,512)     #==>(16, 512, 512)
        x.append(dcm)                     #==>(100, 16, 512, 512)
        y.append(classfic-1)                #==>(100,1)
    return x, y

def ImageMosaic(CTimage):
    newimage = np.reshape(CTimage,(4,4,512,512))
    newimage = newimage.numpy().transpose((0,2,1,3))
    newimage1 = np.reshape(newimage,(2048,4,512))
    finalimage = np.reshape(newimage1,(2048,2048))
    # plt.imsave('test.png', finalimage)
    return finalimage


if __name__ == "__main__":
    # x, y = Generations(3000)
    # torch.save(x,'datax.npy')
    # torch.save(y,'datay.npy')
    # print(mat)
    #尺度变化
    fractal2 = np.load("circle.npy")  # 圆
    fractal3 = np.load("triange.npy")  # 三角
    fractal4 = np.load("squaretan.npy")  # 方
    for r in [80]:
        if r == 80:
            num = 4
        elif r == 60:
            num = 3
        elif r == 40:
            num = 4
        mask = np.random.normal(size=(224, 224), scale=0, loc=1)
        image_circle = unit_circle(224, r, mask,num)
        image_circle = np.multiply(fractal2, image_circle)
        image_rot_circle = unit_rot(image_circle)
        image_square = unit_square(224, r, mask,num)
        image_square = np.multiply(fractal4, image_square)
        image_rot_square = unit_rot(image_square)
        image_tangle = unit_triangle(224, r, mask,num)
        image_tangle = np.multiply(fractal3, image_tangle)
        image_rot_tangle = unit_rot(image_tangle)

        Image.fromarray((image_circle * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testcircle.png')
        Image.fromarray((image_square * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testsquare.png')
        Image.fromarray((image_tangle * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testangle.png')
        Image.fromarray((image_rot_circle * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testcircle90.png')
        Image.fromarray((image_rot_square * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testsquare90.png')
        Image.fromarray((image_rot_tangle * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testtangle90.png')
        Image.fromarray((image_rot_tangle * 255).astype('uint8'), mode='L').convert('RGB').save(str(r)+'frac'+'testtangle90.png')

