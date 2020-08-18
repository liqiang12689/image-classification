import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
import torch

def unit_circle(d,r,classfic=1):
    '''
    d:图像边长
    r:蒙色直径
    '''
    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    mat = np.zeros((d, d))
    mask = np.random.rand(d,d)*classfic
    rx , ry = int(d/2), int(d/2)
    for row in range(d):
        for col in range(d):
            dist = distance(rx, ry, row, col)
            # print((row,col),dist)
            if abs(dist) < r:
                mat[row, col] = 1
    finalmat = np.multiply(mat,mask)
    plt.imsave('test.png',finalmat)
    return finalmat

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
        # print('disr:',disr)
        # print('r:',r)
    dcm = dcm.reshape(16,1,512,512)
    dcm = torch.from_numpy(dcm)
    img = torchvision.utils.make_grid(dcm,nrow=4,normalize=True)
    img = img.numpy().transpose((1,2,0))
    plt.imsave('test.png',img)
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
    x, y = Generations(3000)
    torch.save(x,'datax.npy')
    torch.save(y,'datay.npy')
    # print(mat)