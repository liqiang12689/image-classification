import numpy as np
import matplotlib.pyplot as plt
import xlrd
from brokenaxes import brokenaxes
import os
#读取excel文件
def excel():
    # 打开文件
    workBook = xlrd.open_workbook('./data.xlsx')

    # 1.获取sheet的名字
    # 1.1 获取所有sheet的名字(list类型)
    allSheetNames = workBook.sheet_names()
    print(allSheetNames)

    # # 1.2 按索引号获取sheet的名字（string类型）
    # sheet1Name = workBook.sheet_names()[0]
    # print(sheet1Name)
    ## 2.2 法2：按sheet名字获取sheet内容
    all_loss=[]
    all_acc =[]
    for sheetname in allSheetNames:
        sheet_content = workBook.sheet_by_name(sheetname)

        # 3. sheet的名称，行数，列数
        print(sheet_content.name, sheet_content.nrows, sheet_content.ncols)
        # 4. 获取整行和整列的值（数组）
        experiment1_loss = []
        experiment1_acc = []
        for col in range(9):
            loss = sheet_content.col_values(4 * col + 1)  # 获取第三列内容
            acc = sheet_content.col_values(4 * col + 2)  # 获取第三列内容
            experiment1_loss.append(loss)
            experiment1_acc.append(acc)
        all_loss.append(experiment1_loss)
        all_acc.append(experiment1_acc)
    # print(all_loss)#11*9*301->loss
    # print(all_acc)#11*9*301->acc
    return all_loss,all_acc

def pltfig(all_loss,all_acc):
    netlist = ['mobilenet', 'resnet', 'shufflenet', 'squeezenet', 'alexnet', 'densenet', 'googlenet', 'MNASNet',
               'VGG']
    ##### 不同网络间的图像
    id = 0
    for temploss in all_loss:
        id += 1
        ### 画loss图
        plt.figure()
        # for j in range(len(temploss)):
        #     plt.plot(list(range(300)),np.array(list(map(float,temploss[j][1:]))))
        #     # plt.draw()
        bax = brokenaxes(ylims=((-0.001, .04), (.06, .07)), hspace=.05, despine=False)
        for j in range(len(temploss)):
            # plt.plot(range(0,len(Alllos[i])), Alllos[i], label=netlist[i])
            # plt.legend()
            bax.plot(list(range(300)), np.array(list(map(float,temploss[j][1:]))), label=netlist[j])
            bax.legend()
        # plt.xlabel('Loss vs. iters')
        # plt.ylabel('Loss')
        bax.set_xlabel('Loss vs. iters',labelpad=2)
        bax.set_ylabel('Loss')
        plt.savefig(
            os.path.join('/media/liqiang/windata/project/classification/plugin/newresult', 'ex' +str(id) + 'train_' + "loss.jpg"))
        plt.close()
        ### 画acc图
    ida = 0
    for tempacc in all_acc:
        ida += 1
    ### 画loss图
        plt.figure()
        # for j in range(len(temploss)):
        #     plt.plot(list(range(300)),np.array(list(map(float,temploss[j][1:]))))
        #     # plt.draw()
        for j in range(len(tempacc)):
            # plt.plot(range(0,len(Alllos[i])), Alllos[i], label=netlist[i])
            # plt.legend()
            plt.plot(list(range(300)), np.array(list(map(float, tempacc[j][1:]))), label=netlist[j])
            plt.legend()
        # plt.xlabel('Loss vs. iters')
        # plt.ylabel('Loss')
        plt.xlabel('Accuracy vs. iters')
        plt.ylabel('Accuracy')
        plt.savefig(
            os.path.join('/media/liqiang/windata/project/classification/plugin/newresult',
                         'ex' + str(ida) + 'train_' + "acc.jpg"))
        plt.close()
    #### 同一网络图像
    ### 画loss图
    for m in range(9):
        plt.figure()
        k = 0
        for temploss in all_loss:
            k += 1
            # for j in range(len(temploss)):
            #     plt.plot(list(range(300)),np.array(list(map(float,temploss[j][1:]))))
            #     # plt.draw()
            plt.plot(list(range(300)), np.array(list(map(float, temploss[m][1:]))), label='exp'+str(k))
            plt.legend(loc='upper right')
            # plt.xlabel('Loss vs. iters')
            # plt.ylabel('Loss')
        plt.xlabel('Loss vs. iters')
        plt.ylabel('Loss')
        plt.title(netlist[m])
        plt.savefig(
            os.path.join('/media/liqiang/windata/project/classification/plugin/newresult', netlist[m] + '_train_' + "loss.jpg"))
        plt.close()
    ### acc
    for n in range(9):
        plt.figure()
        l = 0
        for tempacc in all_acc:
            l += 1
            plt.plot(list(range(300)), np.array(list(map(float, tempacc[n][1:]))), label='exp' + str(l))
            plt.legend()
            # plt.xlabel('Loss vs. iters')
            # plt.ylabel('Loss')
        plt.xlabel('Accuracy vs. iters')
        plt.ylabel('Accuracy')
        plt.title(netlist[n])
        plt.savefig(
            os.path.join('/media/liqiang/windata/project/classification/plugin/newresult',
                         netlist[n] + '_train_' + "acc.jpg"))
        plt.close()
if __name__ == "__main__":

    all_loss,all_acc = excel()

    pltfig(all_loss,all_acc)