#  Pytorch 0.4.0 VGG16实现cifar10分类.  
# @Time: 2018/6/23
# @Author: xfLi
'''
本程序实现生成测试矩阵（图像），来测试网络的可行性
测试数据200*16*512*512
bathsize:20
测试方法：每次输入的图像为相同位置的图像即每套图像第一张或者第n张，
该网络测试相同位置图像的分类能力
'''
#前三行解决自定义包报错问题
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import numpy as np
# from tensorboardX import SummaryWriter
from timeplot import epochplt
import cv2
from torchvision import datasets,transforms, models
import torchvision
import modelnet
from plugin.processplugin import Generations
from torch.autograd import Variable
import random
from shapedata import *
import matplotlib.pyplot as plt
import scipy.sparse as sp

from pygcn.coarsening import normalize, adjacency
import torch.nn.functional as F
# model_path = './model_pth/vgg16_bn-6c64b313.pth'



def _next_batch(train_labels, batch_size, index_in_epoch, mask):
    start = index_in_epoch
    index_in_epoch += batch_size
    num_examples = train_labels.__len__()
    train_images = []
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        # perm = np.arange(num_examples)
        # np.random.shuffle(perm)
        # train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    labels = train_labels[start:end]
    for i in labels:
        if i==0:
            img = unit_circle(224, 80, mask)
        elif i==1:
            img = unit_square(224,80,mask)
        elif i==2:
            img = unit_triangle(224,100,mask)
        train_images.append(img)

    return train_images, train_labels[start:end], index_in_epoch


def Scramdata(train_x, train_y):
    """
    Scrambling data order
    :param data:
    :return:
    """
    data_num = len(train_x)
    data_idex = [i for i in range(data_num)]
    random.shuffle(data_idex)
    newtrain_x = [train_x[idx] for idx in data_idex]
    newtrain_y = [train_y[idx] for idx in data_idex]
    return newtrain_x, newtrain_y


def onehotLab(batch_ys, type=False):
    labels = torch.empty(batch_size, dtype=torch.long)
    for lab, tempy in enumerate(batch_ys):
        labels[lab] = torch.tensor(int(tempy))
    if type == True:
        one_hot = torch.zeros(2, 5).scatter_(1, labels, 1)
    else:
        one_hot = labels
    return one_hot
def getpaths(path):
    paths = []
    for tempfile in os.listdir(picPath):
        paths.append(os.path.join(path,tempfile))
    return paths
def train():
    # net = alexnet()
    # print(net)
    # use_gpu = True
    
    # if use_gpu:
    #     net = net.cuda()
    # x, y = Generations(200)
     
    torch_device = torch.device('cuda')

    # train_x, train_y, test_x, test_y, val_x, val_y = getData()
    #load net
    layer = 1   #channels
    use_gpu = True #是否使用gpu
    pretrained = False #是否使用与训练模型
    batch_size = 6
    netname = 'GCNnet'
    if netname=='GCNnet':
        net = modelnet.GCNnet(use_gpu)
    print(net)
    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    # optimizer = torch.optim.Adam(net.classifier.parameters())
    # Train the model
    y0 = np.zeros(1000,dtype=np.int)
    y1 = np.ones(1000, dtype=np.int)
    y2 = np.ones(1000, dtype=np.int)*2
    y_label = np.concatenate((y0,y1,y2),axis=0)
    scale = 1
    loc   = 0
    mask = np.random.normal(size=(224,224), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
    ##生成测试图像
    maxacc = []
    Accuracy_list = []
    Loss_list     = []
    for epoch in range(1):
        # optimizer = torch.optim.Adam(net.parameters())
        #打乱数据和标签
        num = random.randint(1,2000)
        random.seed(num)
        random.shuffle(y_label)
        index_in_epoch = 0
        running_loss = 0.0
        running_correct = 0
        batch = 0
        for iters in range(int(y_label.__len__()/batch_size)):
            batch += 1
            batch_x, batch_y, index_in_epoch = _next_batch(y_label, batch_size, index_in_epoch,mask)
            # for step, (inputs, labels) in enumerate(trainset_loader):
            # batch_xs = preprocess(batch_xs,layer)
            # batch_x = np.array([t.numpy() for t in batch_xs])
            # optimizer.zero_grad()  # 梯度清零                
            labels = batch_y.copy() 
            # tempdata = np.reshape(batch_x,(batch_size, 1, 224, 224))
            # batch_xx = torch.tensor(tempdata, dtype=torch.float)
            if use_gpu==True:
                # batch_xx = batch_xx.to(torch_device)
                batch_xx,labels = Variable(torch.tensor(batch_x).cuda()), Variable(torch.tensor(labels).cuda())
            else:
                batch_xx,labels = Variable(batch_x), Variable(labels)
            adj = adjacency(labels)
            features = torch.reshape(batch_xx, [6, 224*224])
            features = sp.csr_matrix(features)
            print('features_size:', features.shape)
            features = normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            optimizer.zero_grad()
            output = net(features,adj)
            if netname=='googlenet':
                output = output.logits
            _,pred = torch.max(output.data, 1)
            # loss = criterion(output, onehotLab(labels, False))
            loss = criterion(output, labels)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()                
            running_loss += loss.data
            # running_loss += loss.item()
            running_correct += torch.sum(pred == labels)      
            if running_correct.item()/(batch_size*batch) > 0:
                print("Batch {}, Train Loss:{:.6f}, Train ACC:{:.4f}".format(
                batch, running_loss/(batch_size*batch), running_correct.item()/(batch_size*batch)))
                # print('预测标签：{}, 真实标签：{}'.format(pred, labels))
            maxacc.append(running_correct.item()/(batch_size*batch))
            Accuracy_list.append(running_correct.item()/(batch_size*batch))
            Loss_list.append(running_loss/(batch_size*batch))
        print('maxacc:',np.max(maxacc))
        print('预测标签：{}, 真实标签：{}'.format(pred, labels))
        # print('学习率LR:', 5e-03/(5*(epoch+1)))
        x1 = range(0, 500)
        x2 = range(0, 500)
        y1 = Accuracy_list
        y2 = Loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Accuracy vs. iters')
        plt.ylabel('Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Loss vs. iters')
        plt.ylabel('Loss')        
        plt.savefig(netname+'_'+"accuracy_loss.jpg")
        plt.show()



if __name__ == '__main__':
    net = train()