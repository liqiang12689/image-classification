import numpy as np
import scipy
from sklearn import metrics
from numpy import random
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster, preprocessing, decomposition
from scipy.sparse import csgraph
import  torchvision,torch
import  torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from tensorflow.keras import layers
#############################################################
#将每个像素点作为graph中的结点，然后随机采样400个结点，每个结点连接最近的8个结点，生成对应的graph
def generate_grid_graph(m, k=8, num_samples=400):
    #生成m*m的图像中每个像素点的坐标
    M = m**2
    x = np.linspace(0, 1, m, dtype=np.float32)
    y = np.linspace(0, 1, m, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    
    # 随机选择部分结点
    sample_inds = random.choice(range(M), num_samples, False)
    sample_inds.sort()
    z_sample = z[sample_inds, :]
    
    #计算像素点之间的距离
    dist = metrics.pairwise_distances(
        z_sample, metric='euclidean')
    #取前k个最近邻结点 k-NN graph.
    idx = np.argsort(dist)[:, 1:k+1] #前k个最近邻结点的idx
    dist.sort()
    dist = dist[:, 1:k+1] #前k个最近邻结点的距离
    
    # 根据距离计算边的权重
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # 权重矩阵
    I = np.arange(0, num_samples).repeat(k)
    J = idx.reshape(num_samples*k)
    V = dist.reshape(num_samples*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(num_samples, num_samples))

    # 去除自连接
    W.setdiag(0)

    # 数值计算问题，引起的矩阵不对称，修正这一问题
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    
    return sample_inds, W

if __name__ == "__main__":
    img_width = 28
    sample_inds, W_sample = generate_grid_graph(28)
    # 读取MNIST数据
    train = torchvision.datasets.MNIST(root='./mnist/',train=True, transform= transforms.ToTensor(),download=True)
    dataloader = DataLoader(train, batch_size=50,shuffle=True, num_workers=4)
    (x_train, y_train) = train.train_data,train.train_labels    
    x_train_sample = x_train.reshape(x_train.shape[0], -1, 1)[:, sample_inds]/255.0
    f, axes = plt.subplots(1, 2)
    # 可视化采样后的图像
    minist_img = x_train[0]
    mask = np.zeros_like(minist_img).reshape(-1)
    mask += 1
    mask[sample_inds] = 0
    minist_img = minist_img.reshape(img_width, img_width)
    mask = mask.astype(np.bool).reshape(img_width, img_width)

    fig1 = sns.heatmap(minist_img, cmap="YlGnBu", ax=axes[0], xticklabels=False, yticklabels=False, cbar=False, square=True).set_title("original image")
    fig2 = sns.heatmap(minist_img, cmap="YlGnBu", ax=axes[1], xticklabels=False, yticklabels=False, cbar=False, square=True).set_title("subsample image")
    plt.show()