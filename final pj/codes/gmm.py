import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.stats import norm
import matplotlib
import time
font = {'family' : 'SimHei'}
matplotlib.rc('font', **font)

def GMM(img, K, threshold, mu,sigma):
    # 将数据展开成向量
    X = img.reshape(-1)
    # 像素个数
    N = len(X)
    # 记录每个像素在每一类的概率的矩阵
    Pmatrix = np.ones((N, K)) / K
    # 参数pi, mu和sigma已经在参数中赋值
    pi = [1 / K] * K
    while True:
        # 保留上一次的均值向量
        last_mu = mu
        # 更新后验概率矩阵
        for each_class in range(K):
            Pmatrix[:, each_class] = pi[each_class] * norm.pdf(X, mu[each_class], sigma[each_class])
        Pmatrix = Pmatrix / Pmatrix.sum(axis=1).reshape(-1,1)

        # 更新参数pi
        pi = Pmatrix.sum(axis=0) / N

        # 更新参数mu
        mu = np.zeros(K)
        for each_class in range(K):
            mu[each_class] = np.average(X, axis=0, weights=Pmatrix[:, each_class])
        if np.sum((last_mu - mu) ** 2) <= threshold:
            break # 如果两次均值变化很小就不再更新了

        # 更新参数sigma
        for each_class in range(K):
            sigma[each_class] = np.sqrt(np.average((X - mu[each_class]) ** 2, axis=0, weights=Pmatrix[:, each_class]))

    newImg = X
    for each_pixel in range(N):
        newImg[each_pixel] = np.argmax(Pmatrix[each_pixel, :]) # 分类图中每个像素的值取概率最高的那一类
    newImg = ((newImg.reshape(img.shape)) / K * 255).astype(np.uint8)
    return newImg
start = time.time()
img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('原图')
plt.axis('off')
plt.subplot(1,2,2)
plt.axis('off')
#GMMImg = GMM(img,K=2, threshold=1, mu = np.array([10,90]),sigma = np.array([80,80]))
#GMMImg = GMM(img,K=3, threshold=1, mu = np.array([20,100, 170]),sigma = np.array([70]*3))
GMMImg = GMM(img,K=4, threshold=1, mu = np.array([20,80,140,200]),sigma = np.array([60]*4))
plt.title('GMM结果')
plt.imshow(GMMImg, cmap='gray')
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))
plt.show()
