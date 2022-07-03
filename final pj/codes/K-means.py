import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def k_means(image,k=2,iters=1000):
    """
    k均值聚类
    :param image: 输入图像 ndarray
    :param k: 聚类数目 int
    :param iters: 最大迭代次数 int
    :return:
    """
    m, n, z = image.shape
    new_image = np.reshape(image,(m*n,z))  # 排列成m*n行z列
    # 随机选k个聚类中心
    cluster = new_image[random.sample(range(m*n), k), :]
    iter = 0
    tol = 1e-11  # 容忍度
    J_prev = float('inf')
    J = []
    # 迭代
    while True:
        iter = iter + 1
        # 计算每个像素点分别到k个类中心的距离
        dist = np.dot(np.sum(new_image**2,axis=1).reshape((m*n,1)),np.ones((1,k))) + \
            np.dot(np.sum(cluster**2,axis=1).reshape((k,1)), np.ones((1, m*n))).T - 2*np.dot(new_image, cluster.T)
        # 给每个像素点分类
        label = np.argmin(dist,axis=1)
        # 取新的k个聚类中心
        for i in range(k):
            cluster[i,:] = np.mean(new_image[label==i,:], axis=0)
        # 记录目标函数
        J_cur = np.sum((new_image - cluster[label,:])**2)
        J.append(J_cur)
        print('iteration:{0},object fucntion:{1}'.format(iter,J_cur))
        # 当目标函数不再变化时，停止迭代
        if np.abs(J_cur-J_prev) < tol:
            break
        if iter == 1000:
            break
        J_prev = J_cur
    return cluster[label,:].reshape((m,n,z))

if __name__ == "__main__":
    # 原图
    im = Image.open('Lenna.jpg').convert('L')
    im_ary = np.array(im, dtype='int32')
    im_ary = np.expand_dims(im_ary, axis=2)

    new_ary = k_means(im_ary, k=4)
    new_ary = np.squeeze(new_ary, axis=2)
    plt.imshow(new_ary, cmap='gray')
    plt.axis("off")
    plt.savefig('K-means_2')