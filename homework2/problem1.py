from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
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

#BGT算法
def BGT(imhist, epsilon = 0.1):
    L = imhist.shape[0]
    N = np.sum(imhist)
    standn = np.arange(L)
    mu_g = np.sum(standn*imhist)/N   #图片的全局像素平均值
    T = int(mu_g)
    while True:
        R_1 = imhist[:T]/N
        R_2 = imhist[T:]/N
        m_1 = np.sum(R_1)
        m_2 = 1 - m_1
        mu_1 = np.sum(np.arange(T)*R_1)/m_1
        mu_2 = np.sum(np.arange(T,L)*R_2)/m_2
        new_T = int((mu_1+mu_2)/2)
        if abs(new_T - T) < epsilon:
            break
        T = new_T
    return T

if __name__ == '__main__':
    im = Image.open('polymersomes.tif').convert('L')
    im_ary = np.array(im, dtype='int32')
    new_image = plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    m,n = im_ary.shape
    N = m*n   #总像素个数
    L = 256
    imhist = nD_histogram(im_ary.reshape((N, 1)), 1, [L], [0], [L])  # 产生图片的直方图
    T = BGT(imhist)
    im_ary[im_ary <= T] = 0
    im_ary[im_ary > T] = 1
    new_imarry = im_ary * (L - 1)
    plt.imshow(new_imarry, cmap='gray')
    plt.axis("off")
    plt.show()