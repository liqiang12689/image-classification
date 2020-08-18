import numpy as np
import matplotlib.pyplot as plt
import xlrd
from brokenaxes import brokenaxes
import os
from testplt import excel
all_loss,all_acc = excel()
all_acc_mean = []
for i in range(len(all_acc)):
    acc_meam = []
    for j in range(len(all_acc[i])):
        m = np.mean(np.array(list(map(float,all_acc[i][j][101:]))))
        acc_meam.append(m)
        print(m)
    all_acc_mean.append(acc_meam)

