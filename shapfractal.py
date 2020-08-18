import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch

import itertools
import os
def unit_circle(d,r,mask):
    '''
    生成圆形
    d:图像边长
    r:蒙色直径
    '''
    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    mat1 = np.zeros((d, d))
    mat2 = np.zeros((d, d))
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
    mat2 = unit_gauss(d)
    finalmat = np.multiply(mat1,mask)
    finalmat = np.multiply(mat2,finalmat)
    # plt.imsave('testcircle.png',finalmat)
    # Image.fromarray((finalmat * 255).astype('uint8'), mode='L').convert('RGB').save('testcircle.png')
    return finalmat

def unit_square(d,r,mask):
    '''
    生成正方形
    d:图像边长
    r:蒙色直径
    '''
    def distance(x1, y1, x2, y2):
        return abs(x1 - x2)
    mat1 = np.zeros((d, d))
    mat2 = np.zeros((d, d))
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
    mat2 = unit_gauss(d)
    finalmat = np.multiply(mat1,mask)
    finalmat = np.multiply(mat2,finalmat)
    # plt.imsave('testsquare.png',finalmat)
    # Image.fromarray((finalmat * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare.png')
    return finalmat
def unit_triangle(d,r,mask):
    '''
    生成等边三角形
    d:图像边长
    r:蒙色直径
    '''
    mat1 = np.zeros((d, d))
    mat2 = np.zeros((d, d))
    # mask = np.random.normal(size=(d,d), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
    sd = 2*r
    for i in range((d-r)//2,(d-r)//2+r):
        for j in range((i-(d-r)//2+1)*2):
            mat1[i,d//2-(i-(d-r)//2+1)+j] = 1
            # if (i)%20>14:
            #     mat2[i, d//2-(i-(d-r)//2+1)+j] = 1
    # 渐变黑白高斯纹理
    mat2 = unit_gauss(d)
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

def unit_gauss(N):
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)
    X,Y = np.meshgrid(x,y) #生成一个矩阵，并填充数据.
    Z = np.sin(0.5 * np.pi * Y)
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

def fractalmodel(sidechief):
    # 方块分形
    import pygame
    maxlen = sidechief  # 边界
    pygame.init()
    screen = pygame.display.set_caption('方块分形')
    screen = pygame.display.set_mode([maxlen, maxlen])
    screen.fill([255, 255, 255])
    pygame.display.flip()

    def draw(st, leni):
        # st: 左上角点位置[left,top]
        # leni: 当前方块边长
        if leni > 3:
            leni /= 3
            draw(st, leni)  # 左上
            draw([st[0] + leni * 2, st[1]], leni)  # 右上
            draw([st[0] + leni, st[1] + leni], leni)  # 中间
            draw([st[0], st[1] + leni * 2], leni)  # 左下
            draw([st[0] + leni * 2, st[1] + leni * 2], leni)  # 右下
            pygame.display.flip()
        else:
            pygame.draw.rect(screen, [0, 0, 0], [st[0], st[1], leni, leni])

    draw([0, 0], maxlen)
    #获取图形矩阵
    array = pygame.surfarray.array2d(screen)
    #二值话
    array[array>0] = 1
    array[array<1] = 0
    ##保存方形分型
    np.save("square.npy", array)
    # while 1:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             exit()
    #################################################################
    # 圆形分形
    def frac_circle(mat1, r, cx, cy):
        '''
        生成圆形
        r:半径
        cx,cy分别是中心点坐标
        '''

        def distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        rx, ry = int(cx), int(cy)
        for row in range(cx - r, cx + r):
            for col in range(cy - r, cy + r):
                dist = distance(rx, ry, row, col)
                # print((row,col),dist)
                if abs(dist) > r - 1 and abs(dist) < r + 1:
                    mat1[row, col] = 1
        finalmat = mat1
        return finalmat

    def points(p, leni):
        # 返回p,leni对应的半径和圆心列表
        return [(p[0] - leni // 2, p[1] - leni // 2), (p[0] - leni // 2, p[1] + leni // 2),
                (p[0] + leni // 2, p[1] - leni // 2), (p[0] + leni // 2, p[1] + leni // 2)], int(leni // 2)

    def fracircle(mat1, p, leni):
        '''
        生成图形大小
        '''
        mat1 = frac_circle(mat1, int(leni // 2), p[0], p[1])
        certen, r = points(p, leni)[0], int(points(p, leni)[1])
        for i in range(4):
            cx, cy = int(certen[i][0]), int(certen[i][1])
            if leni > 2:
                mat1 = fracircle(mat1, (cx, cy), r)
        # if leni > 2:
        #     mat1 = fracircle(mat1,p,leni)
        leni /= 2
        leni = int(leni)

        return mat1

    d = sidechief
    mat1 = np.zeros((d, d))
    array = fracircle(mat1, (d // 2, d // 2), d // 2)
    # 二值话
    array[array > 0] = 1
    array[array < 1] = 0
    ##保存方形分型
    np.save("circle.npy", array)
    #################################################################
    # 康托尘埃

    pygame.init()
    screen = pygame.display.set_caption('康托尘埃')
    screen = pygame.display.set_mode([sidechief, sidechief])
    screen.fill([255, 255, 255])
    pygame.display.flip()

    cantor = [1, ]  # 起点集，最小像素为1

    while (cantor[-1] + 1) * 3 < 1000:
        st = (cantor[-1] + 1) * 2  # 下一迭代起点
        tep = []
        for i in cantor:
            tep.append(st + i)  # 重复上一子集
        cantor.extend(tep)
    # print(cantor[-1]) # 输出最大像素起点
    for i in cantor:
        for j in cantor:
            screen.set_at([i, j], [0, 0, 0])
    pygame.display.flip()
    # 获取图形矩阵
    array = pygame.surfarray.array2d(screen)
    # 二值话
    array[array > 0] = 1
    array[array < 1] = 0
    ##保存方形分型
    np.save("kangtuo.npy", array)
    # while 1:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             exit()
    # 谢尔宾斯基三角垫

    # maxlen = 500  # 边界

    pygame.init()
    screen = pygame.display.set_caption('谢尔宾斯基三角垫')
    screen = pygame.display.set_mode([sidechief, sidechief])
    screen.fill([255, 255, 255])
    pygame.display.flip()

    def mid(a, b):
        # 求出a, b点的中点坐标
        return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

    def draw(one, two, tri):
        # 参数代表三个顶点,上、左、右排序
        if one[0] - two[0] > 2:  # 可分
            draw(one, mid(one, two), mid(one, tri))  # 画上面的三角
            draw(mid(one, two), two, mid(two, tri))  # 画左边三角
            draw(mid(one, tri), mid(two, tri), tri)  # 画右边的三角
            pygame.display.flip()
        else:  # 达到最小结构
            pygame.draw.polygon(screen, [0, 0, 0], [one, two, tri])

    draw([maxlen / 2, 0], [0, maxlen], [maxlen, maxlen])
    # 获取图形矩阵
    array = pygame.surfarray.array2d(screen)
    # 二值话
    array[array > 0] = 1
    array[array < 1] = 0
    array = unit_rot(array)
    array = unit_rot(array)
    array = unit_rot(array)
    ##保存方形分型
    np.save("triange.npy", array)
    # while 1:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             exit()
    ############################################################
    # 谢尔宾斯基方毯
    pygame.image.save(screen, 'test.png')


    # maxlen = 500  # 边界

    pygame.init()
    screen = pygame.display.set_caption('谢尔宾斯基方毯')
    screen = pygame.display.set_mode([sidechief, sidechief])
    screen.fill([0, 0, 0])
    pygame.display.flip()

    def p2(p, r, d):
        # p: 参考左上顶点
        # r: 距离参考点向右偏移距离
        # d: 距离参考点向下偏离距离
        return [p[0] + r, p[1] + d]

    def points(p, leni):
        # 返回p,leni对应的四边形四个顶点列表
        return [p, p2(p, leni, 0), p2(p, leni, leni), p2(p, 0, leni)]

    def draw(p, leni):
        # p：左上顶点
        # leni：边长
        leni /= 3
        pygame.draw.polygon(screen, [255, 255, 255],
                            points(p2(p, leni, leni), leni))
        if leni > 3:
            draw(p, leni)
            draw(p2(p, leni, 0), leni)
            draw(p2(p, 2 * leni, 0), leni)

            draw(p2(p, 0, leni), leni)
            draw(p2(p, 2 * leni, leni), leni)

            draw(p2(p, 0, 2 * leni), leni)
            draw(p2(p, leni, 2 * leni), leni)
            draw(p2(p, 2 * leni, 2 * leni), leni)
            pygame.display.flip()

    draw([0, 0], maxlen)
    # 获取图形矩阵
    array = pygame.surfarray.array2d(screen)
    # 二值话
    array[array > 0] = 1
    array[array < 1] = 0
    ##保存方形分型
    np.save("squaretan.npy", array)


if __name__ == "__main__":
    # x, y = Generations(3000)
    # torch.save(x,'datax.npy')
    # torch.save(y,'datay.npy')
    # print(mat)
    mask = np.random.normal(size=(224, 224), scale=0, loc=1)
    image_circle = unit_circle(224, 80, mask)
    image_rot_circle = unit_rot(image_circle)
    image_square = unit_square(224, 80, mask)
    image_rot_square = unit_rot(image_square)
    image_tangle = unit_triangle(224, 80, mask)
    image_rot_tangle = unit_rot(image_tangle)
    ##进行分型处理
    #1）生成分型图案模体
    fractalmodel(224)
    #)2）加载图案模体
    fractal1 = np.load('square.npy')
    fractal2 = np.load("circle.npy")#圆
    fractal3 = np.load("triange.npy")#三角
    fractal4 = np.load("squaretan.npy")#方
    '''
    plt.matshow(fractal1)
    plt.matshow(fractal2)
    plt.matshow(fractal3)
    plt.matshow(fractal4)



    Image.fromarray((image_circle * 255).astype('uint8'), mode='L').convert('RGB').save('testcircle.png')
    Image.fromarray((image_square * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare.png')
    Image.fromarray((image_tangle * 255).astype('uint8'), mode='L').convert('RGB').save('testangle.png')
    Image.fromarray((image_rot_circle * 255).astype('uint8'), mode='L').convert('RGB').save('testcircle90.png')
    Image.fromarray((image_rot_square * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare90.png')
    Image.fromarray((image_rot_tangle * 255).astype('uint8'), mode='L').convert('RGB').save('testtangle90.png')

    Image.fromarray((fractal1 * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare.png')
    Image.fromarray((fractal1 * 255).astype('uint8'), mode='L').convert('RGB').save('testsquare.png')
    Image.fromarray((fractal3 * 255).astype('uint8'), mode='L').convert('RGB').save('testtriange.png')
    Image.fromarray((fractal4 * 255).astype('uint8'), mode='L').convert('RGB').save('testsquaretan.png')
    '''
