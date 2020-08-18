#  Pytorch 0.4.0 VGG16实现cifar10分类.  
# @Time: 2018/6/23
# @Author: xfLi
'''
本程序实现生成测试矩阵（图像），来测试网络的可行性
测试数据尺寸：224*224
bathsize:20
输入图像只含有形状数据和噪声数据
'''
#前三行解决自定义包报错问题

import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
# from tensorboardX import SummaryWriter
# from timeplot import epochplt
import cv2
from torchvision import datasets,transforms, models
import torchvision
import modelnet
# from plugin.processplugin import Generations
from torch.autograd import Variable
import random
from cleardata import *
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
# model_path = './model_pth/vgg16_bn-6c64b313.pth'
import time


def _next_batch(train_labels, batch_size, index_in_epoch, mask):
    start = index_in_epoch
    index_in_epoch += batch_size
    num_examples = train_labels.__len__()
    train_images = []
    rotIdex = False
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    labels = train_labels[start:end]
    newlabels = []
    for i in labels:
        if i==0:
            img = unit_circle(224, 80, mask)
        elif i==1:
            img = unit_square(224,80,mask)
        elif i==2:
            img = unit_triangle(224,100,mask)
        train_images.append(img)
        newlabels.append(i)
        if rotIdex:
            img = unit_rot(img)
            train_images.append(img)
            newlabels.append(i)

    return train_images, newlabels, index_in_epoch


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
    batch_size = 30

    netlist = ['mobilenet','resnet','shufflenet','squeezenet','alexnet','densenet','googlenet','mnastnet','vgg16']
    # netlist = ['mobilenet','resnet','shufflenet','squeezenet','alexnet','densenet','googlenet','mnastnet']
    # netlist = ['mobilenet','resnet','vgg16']
    # netlist = ['googlenet']
    Allacc = []
    Alllos = []
    val_Allacc = []
    val_Alllos = []
    test_Allacc = []
    test_Alllos = []
    for netname in netlist:
        time_start = time.time()
        if netname=='mobilenet':
            net = modelnet.mobilenet(layer,use_gpu,pretrained)
        elif netname=='resnet':
            net = modelnet.resnet(layer,use_gpu,pretrained)
        elif netname=='shufflenet':
            net = modelnet.shufflenet(layer,use_gpu,pretrained)
        elif netname=='squeezenet':
            net = modelnet.squeezenet(layer,use_gpu,pretrained)
        elif netname=='alexnet':
            net = modelnet.alexnet(layer,use_gpu,pretrained)
        elif netname=='densenet':
            net = modelnet.densenet(layer,use_gpu,pretrained)
        elif netname=='googlenet':
            net = modelnet.googlenet(layer,use_gpu,pretrained)
        elif netname=='mnastnet':
            net = modelnet.mnasnet(layer,use_gpu,pretrained)
        elif netname=='vgg16':
            net = modelnet.vgg16(layer,use_gpu,pretrained)
        # print(netname)
        print(net)
        # Loss and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
        # optimizer = torch.optim.Adam(net.classifier.parameters())
        # Train the model
        y0 = np.zeros(3000,dtype=np.int)
        y1 = np.ones(3000, dtype=np.int)
        y2 = np.ones(3000, dtype=np.int)*2
        y_label = np.concatenate((y0,y1,y2),axis=0)
        scale = 0
        loc   = 1
        
        ##生成测试图像
        maxacc = []
        Accuracy_list = []
        Loss_list     = []
        val_Accuracy_list = []
        val_Loss_list     = []
        test_Accuracy_list = []
        test_Loss_list     = []
        tempacc = 0
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
                mask = np.random.normal(size=(224,224), scale=scale, loc=loc) #loc表示均值，scale表示方差，size表示输出的size
                batch_x, batch_y, index_in_epoch = _next_batch(y_label, batch_size, index_in_epoch,mask)
                # for step, (inputs, labels) in enumerate(trainset_loader):
                # batch_xs = preprocess(batch_xs,layer)
                # batch_x = np.array([t.numpy() for t in batch_xs])
                # optimizer.zero_grad()  # 梯度清零                
                labels = batch_y.copy() 
 
                tempdata = np.reshape(batch_x,(batch_size, 1, 224, 224))
                batch_xx = torch.tensor(tempdata, dtype=torch.float)
                if use_gpu==True:
                    # batch_xx = batch_xx.to(torch_device)
                    batch_xx,labels = Variable(torch.tensor(batch_xx).cuda()), Variable(torch.tensor(labels).cuda())
                else:
                    batch_xx,labels = Variable(batch_xx), Variable(labels)
                optimizer.zero_grad()
                output = net(batch_xx)
                if netname=='googlenet':
                    if len(output)==3:
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
        time_end = time.time()
        print('totally cost',time_end-time_start)


if __name__ == '__main__':
    net = train()