#  Pytorch 0.4.0 VGG16实现cifar10分类.  
# @Time: 2018/6/23
# @Author: xfLi
'''
本程序实现生成测试矩阵（图像），来测试网络的可行性
测试数据尺寸：224*224
bathsize:20
输入图像含有高斯条纹
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
# from timeplot import epochplt
import cv2
from torchvision import datasets,transforms, models
import torchvision
import modelnet
# from plugin.processplugin import Generations
from torch.autograd import Variable
import random
from shapfracdata import *
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
# model_path = './model_pth/vgg16_bn-6c64b313.pth'



def _next_batch(train_labels, batch_size, index_in_epoch, mask,num):
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
    fractal2 = np.load("circle.npy")  # 圆
    fractal3 = np.load("triange.npy")  # 三角
    fractal4 = np.load("squaretan.npy")  # 方
    for i in labels:        
        if i==0:
            img = unit_circle(224, 80, mask,num)
            img = np.multiply(fractal2, img)
        elif i==1:
            img = unit_square(224,80,mask,num)
            img = np.multiply(fractal4, img)
        elif i==2:
            img = unit_triangle(224,100,mask,num)
            img = np.multiply(fractal3, img)
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
        scale = 0.2
        loc   = 0
        
        ##生成测试图像
        maxacc = []
        Accuracy_list = []
        Loss_list     = []
        val_Accuracy_list = []
        val_Loss_list     = []
        test_Accuracy_list = []
        test_Loss_list     = []
        tempacc = 0
        num_i = 2
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
                batch_x, batch_y, index_in_epoch = _next_batch(y_label, batch_size, index_in_epoch,mask,num_i)
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
        
            '''
            print('#######################  运行验证集   ################')
            val_Accuracy, val_Loss = val_train(net,netname,criterion,mask,batch_size,use_gpu)
            #更新精度并保存模型
            if val_Accuracy - tempacc > 0:
                tempacc = val_Accuracy
                torch.save(net,os.path.join('/media/liqiang/windata/project/classification/plugin/model',netname+'_'+'net.pkl'))
            val_Accuracy_list.append(val_Accuracy)
            val_Loss_list.append(val_Loss)
            print('#######################  验证集结束   ################')
            print('#######################  运行测试集   ################')
            test_Accuracy, test_Loss = test_train(netname,criterion,mask,batch_size,use_gpu)
            test_Accuracy_list.append(test_Accuracy)
            test_Loss_list.append(test_Loss)      
            print('#######################  测试集结束   ################')  
            '''  
        #保存网络结构
        torch.save(net,os.path.join('/media/liqiang/windata/project/classification/plugin/model','ex7'+netname+'_'+'net.pkl'))        
        print('预测标签：{}, 真实标签：{}'.format(pred, labels))
        y1 = Accuracy_list
        y2 = Loss_list
        Allacc.append(y1)
        Alllos.append(y2)
        val_Allacc.append(val_Accuracy_list)
        val_Alllos.append(val_Loss_list)
        test_Allacc.append(test_Accuracy_list)
        test_Alllos.append(test_Loss_list)
        
    ###保存训练集训练曲线
    for i in range(len(netlist)):
        plt.plot(range(0,len(Allacc[i])), Allacc[i],label=netlist[i])        
        plt.legend()
    plt.xlabel('Accuracy vs. iters')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','ex7'+'train_'+"accuracy.jpg"))
    plt.show()
    plt.close()
    fig = plt.figure()
    bax = brokenaxes(ylims=((-0.001, .04), (.06, .07)), hspace=.05, despine=False)
    for i in range(len(netlist)):
        # plt.plot(range(0,len(Alllos[i])), Alllos[i], label=netlist[i])
        # plt.legend()
        bax.plot(range(0,len(Alllos[i])), Alllos[i], label=netlist[i])
        bax.legend()
    # plt.xlabel('Loss vs. iters')
    # plt.ylabel('Loss')  
    bax.set_xlabel('Loss vs. iters')
    bax.set_ylabel('Loss')
    # plt.yscale('log')
    # plt.ylim([-0.01,0.06])         
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','ex7'+'train_'+"loss.jpg"))
    plt.show()
    plt.close()

    '''
    ###保存验证集训练曲线
    for i in range(len(netlist)):
        plt.plot(range(0,len(val_Allacc[i])), val_Allacc[i],label=netlist[i])        
        plt.legend()
    plt.xlabel('Accuracy vs. iters')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','val_'+"accuracy.jpg"))
    plt.show()
    plt.close()
    for i in range(len(netlist)):
        plt.plot(range(0,len(val_Alllos[i])), val_Alllos[i], label=netlist[i])
        plt.legend()
    plt.xlabel('Loss vs. iters')
    plt.ylabel('Loss')  
    plt.yscale('log')         
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','val_'+"loss.jpg"))
    plt.show()
    plt.close()
    ###保存测试集训练曲线
    for i in range(len(netlist)):
        plt.plot(range(0,len(test_Allacc[i])), test_Allacc[i],label=netlist[i])        
        plt.legend()
    plt.xlabel('Accuracy vs. iters')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','test_'+"accuracy.jpg"))
    plt.show()
    plt.close()
    for i in range(len(netlist)):
        plt.plot(range(0,len(test_Alllos[i])), test_Alllos[i], label=netlist[i])
        plt.legend()
    plt.xlabel('Loss vs. iters')
    plt.ylabel('Loss')  
    plt.yscale('log')         
    plt.savefig(os.path.join('/media/liqiang/windata/project/classification/plugin/result','test_'+"loss.jpg"))
    plt.show()
    plt.close()
    '''



def val_train(net,netname,criterion,mask,batch_size=30,use_gpu=True):
    ##验证集训练
    
    #1）生成验证集
    val_y0 = np.zeros(300,dtype=np.int)
    val_y1 = np.ones(300, dtype=np.int)
    val_y2 = np.ones(300, dtype=np.int)*2
    val_y_label = np.concatenate((val_y0,val_y1,val_y2),axis=0)
    #2)打乱数据
    val_num = random.randint(1,2000)
    random.seed(val_num)
    random.shuffle(val_y_label)
    val_batch = 0
    val_index_in_epoch = 0
    # val_Accuracy_list = []
    # val_Loss_list = []
    val_running_loss = 0.0
    val_running_correct = 0
    for val_iters in range(int(val_y_label.__len__()/batch_size)):
        val_batch += 1
        val_batch_x, val_batch_y, val_index_in_epoch = _next_batch(val_y_label, batch_size, val_index_in_epoch,mask)
        # for step, (inputs, labels) in enumerate(trainset_loader):
        # batch_xs = preprocess(batch_xs,layer)
        # batch_x = np.array([t.numpy() for t in batch_xs])
        # optimizer.zero_grad()  # 梯度清零                
        val_labels = val_batch_y.copy() 

        val_tempdata = np.reshape(val_batch_x,(batch_size, 1, 224, 224))
        val_batch_xx = torch.tensor(val_tempdata, dtype=torch.float)
        if use_gpu==True:
            # batch_xx = batch_xx.to(torch_device)
            val_batch_xx,val_labels = Variable(torch.tensor(val_batch_xx).cuda()), Variable(torch.tensor(val_labels).cuda())
            net = net.cuda()
        else:
            val_batch_xx,val_labels = Variable(val_batch_xx), Variable(val_labels)
            net = net.copy()
        net.eval()
        val_output = net(val_batch_xx)
        _,val_pred = torch.max(val_output.data, 1)
        val_loss = criterion(val_output, val_labels)
        # loss = loss.requires_grad_()
        # loss.backward()
        # optimizer.step()                
        val_running_loss += val_loss.data
        # running_loss += loss.item()
        val_running_correct += torch.sum(val_pred == val_labels)      
        if val_running_correct.item()/(batch_size*val_batch) > 0:
            print("Batch {}, Train Loss:{:.6f}, Train ACC:{:.4f}".format(
            val_batch, val_running_loss/(batch_size*val_batch), val_running_correct.item()/(batch_size*val_batch)))
        # val_Accuracy_list.append(running_correct.item()/(batch_size*batch))
        # val_Loss_list.append(running_loss/(batch_size*batch))
    return val_running_correct.item()/(batch_size*val_batch), val_running_loss/(batch_size*val_batch)

def test_train(netname,criterion,mask,batch_size=30,use_gpu=True):
    ##验证集训练
    #1）生成验证集
    test_y0 = np.zeros(300,dtype=np.int)
    test_y1 = np.ones(300, dtype=np.int)
    test_y2 = np.ones(300, dtype=np.int)*2
    test_y_label = np.concatenate((test_y0,test_y1,test_y2),axis=0)
    #2)打乱数据
    test_num = random.randint(1,2000)
    random.seed(test_num)
    random.shuffle(test_y_label)
    test_batch = 0
    test_index_in_epoch = 0
    test_running_loss = 0.0
    test_running_correct = 0
    for test_iters in range(int(test_y_label.__len__()/batch_size)):
        test_batch += 1
        test_batch_x, test_batch_y, test_index_in_epoch = _next_batch(test_y_label, batch_size, test_index_in_epoch,mask)
        # for step, (inputs, labels) in enumerate(trainset_loader):
        # batch_xs = preprocess(batch_xs,layer)
        # batch_x = np.array([t.numpy() for t in batch_xs])
        # optimizer.zero_grad()  # 梯度清零                
        test_labels = test_batch_y.copy() 
        test_tempdata = np.reshape(test_batch_x,(batch_size, 1, 224, 224))
        test_batch_xx = torch.tensor(test_tempdata, dtype=torch.float)
        if use_gpu==True:
            # batch_xx = batch_xx.to(torch_device)
            test_batch_xx,test_labels = Variable(torch.tensor(test_batch_xx).cuda()), Variable(torch.tensor(test_labels).cuda())
            net = torch.load(os.path.join('/media/liqiang/windata/project/classification/plugin/model',netname+'_'+'net.pkl'))
            net = net.cuda()
        else:
            test_batch_xx,test_labels = Variable(test_batch_xx), Variable(test_labels)
            net = torch.load(os.path.join('/media/liqiang/windata/project/classification/plugin/model',netname+'_'+'net.pkl'))
        net.eval()
        test_output = net(test_batch_xx)
        _,test_pred = torch.max(test_output.data, 1)
        test_loss = criterion(test_output, test_labels)
        # loss = loss.requires_grad_()
        # loss.backward()
        # optimizer.step()                
        test_running_loss += test_loss.data
        # running_loss += loss.item()
        test_running_correct += torch.sum(test_pred == test_labels)      
        if test_running_correct.item()/(batch_size*test_batch) > 0:
            print("Batch {}, Train Loss:{:.6f}, Train ACC:{:.4f}".format(
            test_batch, test_running_loss/(batch_size*test_batch), test_running_correct.item()/(batch_size*test_batch)))
        # val_Accuracy_list.append(running_correct.item()/(batch_size*batch))
        # val_Loss_list.append(running_loss/(batch_size*batch))
    return test_running_correct.item()/(batch_size*test_batch), test_running_loss/(batch_size*test_batch)
        




if __name__ == '__main__':
    net = train()