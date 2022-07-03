import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def truncation(im_ary,f_min,f_max):
    """
    截断函数
    """
    L = 256
    m,n = im_ary.shape
    for i in range(m):
        for j in range(n):
            if im_ary[i,j] <= f_min:
                im_ary[i,j] = f_min
            if im_ary[i,j] >= f_max:
                im_ary[i,j] = f_max
    return im_ary

def White_noise(shape,a):
    """
    生成白噪声
    input:
    shape:生成白噪声的图片大小
    a:白噪声在频率域的幅度
    """
    L = 256
    m,n = shape
    Wn = np.zeros((m,n,2))
    for i in range(m):
        for j in range(n):
            #随机生成白噪声在频率域中的相位
            theta = 2*np.pi*np.random.rand()
            e_real = a * np.cos(theta)
            e_imag = a * np.sin(theta)
            Wn[i,j,0] = e_real
            Wn[i,j,1] = e_imag
    Wn = cv2.idft(Wn)[:,:,0] #舍去虚部 只取实部
    #归一化后使其成为图像像素值量级
    noise = Wn/np.max(np.abs(Wn))*(L-1)
    noise = noise.astype(np.uint8)
    return noise

def Jiaoyan(shape,a,b,P_a,P_b):
    """
    生成椒盐噪声
    input:
    shape:生成椒盐噪声的图片大小
    a:盐像素值
    P_a:盐概率
    b:椒像素值
    P_b:椒概率
    """
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() <= P_a:
                noise[i,j] = a   #产生盐噪声
            elif np.random.rand() >= 1-P_b:
                noise[i,j] = b   #产生椒噪声
    return noise

if __name__ == '__main__':
    # 白噪声
    #读取图像
    im = Image.open('headCT.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    L = 256
    shape = im_ary.shape
    noise = White_noise(shape,1)
    new_ary = im_ary + noise
    # 对比拉伸
    new_ary = new_ary - np.full(new_ary.shape, np.min(new_ary))
    new_ary = new_ary * (L-1)/ np.max(new_ary)
    new_ary = new_ary.astype(np.uint8)

    plt.subplot(121)
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(new_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 椒盐噪声
    # 读取图像
    im = Image.open('headCT.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    L = 256
    shape = im_ary.shape
    noise = Jiaoyan(shape, 255, -255, 0.1, 0.1)
    new_ary = im_ary + noise
    # 截断
    new_ary = truncation(new_ary, 0, 255)

    plt.subplot(121)
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(new_ary, cmap='gray')
    plt.axis("off")
    plt.show()