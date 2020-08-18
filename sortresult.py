import numpy as np
import matplotlib.pyplot as plt
from testplt import excel
all_loss,all_acc = excel()
## compute mean
'''
all_acc_mean = []
for i in range(len(all_acc)):
    acc_meam = []
    for j in range(len(all_acc[i])):
        m = np.mean(np.array(list(map(float,all_acc[i][j][1:]))))
        acc_meam.append(m)
        print(m)
    all_acc_mean.append(acc_meam)
'''
## compute max acc
all_acc_max = []
for i in range(len(all_acc)):
    acc_max = []
    for j in range(len(all_acc[i])):
        m = np.max(np.array(list(map(float,all_acc[i][j][1:]))))
        acc_max.append(m)
        print(m)
    all_acc_max.append(acc_max)