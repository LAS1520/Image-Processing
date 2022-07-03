import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def Add_SaltAndPepper_Nosie(img, p=0.1):
    """
    添加椒盐噪声
    :param img: 输入图片
    :param p: 噪声产生概率
    :return: 产生噪声的图片
    """
    L = 256
    # 添加盐噪声
    noise = np.random.uniform(0, L - 1, img.shape)
    mask = noise < p * (L - 1)
    img = img * (1 - mask)
    # 添加椒噪声
    mask = noise > (1 - p) * (L - 1)
    img = (L - 1) * mask + img * (1 - mask)
    return img

def k_means_c(image,k=2,iters=1000):
    m, n, z = image.shape
    new_image = np.reshape(image,(m*n,z))  # 排列成m*n行z列
    cluster = new_image[random.sample(range(m*n), k), :]  # 随机选三个聚类中心
    iter = 0
    tol = 1e-11  # 容忍度
    J_prev = float('inf')
    J = []
    while True:
        iter = iter + 1
        dist = np.dot(np.sum(new_image**2,axis=1).reshape((m*n,1)),np.ones((1,k))) + \
            np.dot(np.sum(cluster**2,axis=1).reshape((k,1)), np.ones((1, m*n))).T - 2*np.dot(new_image, cluster.T)
        label = np.argmin(dist,axis=1)
        for i in range(k):
            cluster[i,:] = np.mean(new_image[label==i,:], axis=0)  # 取新的k个聚类中心
        J_cur = np.sum((new_image - cluster[label,:])**2)
        J.append(J_cur)
        print('iteration:{0},object fucntion:{1}'.format(iter,J_cur))
        if np.abs(J_cur-J_prev) < tol:
            break
        if iter == 1000:
            break
        J_prev = J_cur
    return cluster[label,:].reshape((m,n,z))

# 计算直方图
def nD_histogram(data,dimension,nbins,pInMin,pInMax):
    pHistogram = []  # store histogram points
    pHsize = 1
    for idim in range(dimension):
        pHsize *= nbins[idim]
    for i in range(pHsize):
        pHistogram.append(0)
    pBinSpacings = []   # store bin width
    pBinPos = []   # store bin position
    for i in range(dimension):
        pBinSpacings.append(0)
        pBinPos.append(0)
    for idim in range(dimension): #store bin width of different dimensions
        pBinSpacings[idim] = (pInMax[idim] - pInMin[idim])/nbins[idim]
    for idata in range(len(data)):
        for idim in range(dimension):
            value = data[idata][idim]
            pBinPos[idim] = int((value - pInMin[idim])/pBinSpacings[idim])
            #防止越界
            pBinPos[idim] = max(pBinPos[idim],0)
            pBinPos[idim] = min(pBinPos[idim],nbins[idim] - 1)
        index = pBinPos[0]
        for idim in range(1,dimension):
            vSize = 1
            for i in range(idim):
                vSize *= nbins[i]
            index += pBinPos[idim] * vSize
        pHistogram[index] += 1
    return np.array(pHistogram)

#OTSU算法
def OTSU2(imhist):
    L = imhist.shape[0]
    N = np.sum(imhist)
    standn = np.arange(L)
    mu_g = np.sum(standn*imhist)/N   #图片的全局像素平均值
    w_1 = 0
    m_1 = 0
    max_y = -float('inf')
    index = 0
    for i in range(L):  #组内至少有一个元素
        w_1 += imhist[i]/N   # Frequency
        m_1 += i*imhist[i]/N
        if w_1 == 0:
            y = 0
        elif w_1 == 1:
            break
        else:
            y = (mu_g * w_1 - m_1) ** 2 / (w_1 * (1 - w_1))
        if y > max_y:
            max_y = y
            index = i
    return index

if __name__ == '__main__':
     pass