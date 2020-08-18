#  Pytorch 0.4.0 VGG16实现cifar10分类.  
# @Time: 2018/6/23
# @Author: xfLi
#前三行解决自定义包报错问题
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
import numpy as np
from model.alexnet import alexnet
from processdata.getdata import loadata, func, trainSplit
from processdata.readniigz import loadniigz
import random
# from tensorboardX import SummaryWriter
from timeplot import epochplt
from processdata.seglung import savecutlung
import cv2
from torchvision import datasets,transforms, models
import matplotlib.pyplot as plt
import torchvision
from pygcn.models import GCN

def mobilenet(layer,use_gpu,pretrained):
    model = models.mobilenet_v2(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.features[0]   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                        torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        torch.nn.ReLU6(inplace=True))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Dropout(p=0.2, inplace=False),
                                        torch.nn.Linear(in_features=1280, out_features=3, bias=True))

    # for indexf,parmaf in enumerate(model.features[0].parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 1:
    #         parma.requires_grad = True

    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def resnet(layer,use_gpu,pretrained):
    model = models.resnet101(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.conv1   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    model.fc = torch.nn.Sequential(
                                        torch.nn.Linear(in_features=2048, out_features=3, bias=True))

    # for indexf,parmaf in enumerate(model.conv1.parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.fc.parameters()):
    #     if index == 0:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def shufflenet(layer,use_gpu,pretrained):
    model = models.shufflenet_v2_x0_5(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.conv1   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                        torch.nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        torch.nn.ReLU(inplace=True))
    model.fc = torch.nn.Sequential(
                                        torch.nn.Linear(in_features=1024, out_features=3, bias=True))

    # for indexf,parmaf in enumerate(model.conv1.parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.fc.parameters()):
    #     if index == 0:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def squeezenet(layer,use_gpu,pretrained):
    model = models.squeezenet1_0(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.features[0]   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 96, kernel_size=(7, 7), stride=(2, 2)))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Dropout(p=0.5, inplace=False),
                                        torch.nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    # for indexf,parmaf in enumerate(model.features[0].parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 1:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def alexnet(layer,use_gpu,pretrained):
    model = models.alexnet(pretrained = pretrained)
    # print(model)
    # use_gpu = False

    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.features   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer,64,kernel_size=(11,11),stride=(4,4),padding=(2,2)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                                        torch.nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                                        torch.nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Linear(9216, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 3))

    # for indexf,parmaf in enumerate(model.features.parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 6:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def densenet(layer,use_gpu,pretrained):
    model = models.densenet121(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.features.conv0   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Linear(in_features=1024, out_features=3, bias=True))

    # for indexf,parmaf in enumerate(model.features.conv0.parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 0:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def googlenet(layer,use_gpu,pretrained):
    model = models.googlenet(pretrained = pretrained)
    # print(model)
    # use_gpu = False

    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.conv1   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                        torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
    model.aux1.fc2 = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    model.aux2.fc2 = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)

    # for indexf,parmaf in enumerate(model.conv1.parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.fc.parameters()):
    #     if index == 0:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def mnasnet(layer,use_gpu,pretrained):
    model = models.mnasnet0_5(pretrained = pretrained)
    # print(model)
    # use_gpu = False
    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.layers[0]   = torch.nn.Sequential(
                                        torch.nn.Conv2d(layer, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Dropout(p=0.2, inplace=True),
                                        torch.nn.Linear(in_features=1280, out_features=3, bias=True))

    # for indexf,parmaf in enumerate(model.layers[0].parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 1:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def vgg16(layer,use_gpu,pretrained):

    model = models.vgg16(pretrained = pretrained)
    # print(model)
    # use_gpu = False

    for parma in model.parameters():
        parma.requires_grad = False
    # model.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
    model.features[0]   = torch.nn.Sequential(
                                        torch.nn. Conv2d(layer, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    model.classifier = torch.nn.Sequential(
                                        torch.nn.Linear(25088, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 3))

    # for indexf,parmaf in enumerate(model.features[0].parameters()):
    #     if indexf == 0:
    #         parmaf.requires_grad = True

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 6:
    #         parma.requires_grad = True
    
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model

def GCNnet(nfeat=100, nhid=2048, nclass=1000, dropout=0.5,use_gpu = True):
    
    model = GCN(nfeat=100, nhid=2048, nclass=1000, dropout=0.5)
    # print(model)
    # use_gpu = False

    for parma in model.parameters():
        parma.requires_grad = False
        
    if use_gpu:
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model = model.cuda()
    return model
