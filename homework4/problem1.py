import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def Fre_filter(im_ary, H, fre_image=False):
    """
    频率域滤波5步骤
    input:
    im_ary:处理的图像
    H:频率域滤波器
    output:
    new_im:处理之后的图像

    """
    # 第一步，填充图像（padding）
    M, N = im_ary.shape
    new_ary = np.zeros((2 * M, 2 * N))
    # 第二步，计算平移之后的DFT
    for i in range(M):
        for j in range(N):
            new_ary[i, j] = im_ary[i, j] * (-1) ** (i + j)  # 平移
    F = cv2.dft(new_ary, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 第三步，在频率域与滤波器相乘
    G = F * H
    # 画出频谱图
    if fre_image:
        # 原图的频谱图
        f_dft = np.log(cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1)
        plt.subplot(121)
        plt.imshow(f_dft, cmap='gray')
        plt.axis("off")
        # 频域操作结果的频谱图
        g_dft = np.log(cv2.magnitude(G[:, :, 0], G[:, :, 1]) + 1)
        plt.subplot(122)
        plt.imshow(g_dft, cmap='gray')
        plt.axis("off")
        plt.show()
    # 第四步，计算傅里叶逆变换IDFT
    i_dft = cv2.idft(G)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    # 第五步，提取图像
    return result_ary[:M, :N]


def Smoothing(im_ary,ideal=False,Gaussian=False,D_0=60):
    """
    平滑操作
    input:
    im_ary:处理的图像
    ideal:是否使用理想低通滤波器
    Gaussian:是否使用高斯低通滤波器
    D_0:低通范围
    output:
    new_im:处理之后的图像
    """
    M,N = 2*im_ary.shape[0],2*im_ary.shape[1]
    if ideal == True:
        #生成理想低通滤波器
        f = lambda x, y: np.sqrt((x)**2 + (y)**2)
        crow, ccol = int(M / 2), int(N / 2) # 求得图像的中心点位置
        H = np.zeros((M, N, 2))
        for i in range(M):
            for j in range(N):
                distance = f(i-crow,j-ccol)
                if distance <= D_0:
                    H[i,j,0] = 1  #虚部为0
                else:
                    H[i,j,0] = 0
    elif Gaussian == True:
        #生成高斯低通滤波器
        sigma = D_0
        f = lambda x, y: np.exp(-(x**2+y**2)/(2*sigma**2))
        crow, ccol = int(M / 2), int(N / 2) # 求得图像的中心点位置
        H = np.zeros((M, N, 2))
        for i in range(M):
            for j in range(N):
                H[i,j,0] = f(float(i-crow),float(j-ccol))  #虚部为0
    return Fre_filter(im_ary,H,fre_image=True)


def Sharpening(im_ary, Laplace=False, High_boosting=False, D_0=60, k=9):
    """
    锐化操作
    input:
    im_ary:处理的图像
    Laplace:是否使用拉普拉斯算子
    Gaussian:是否使用高斯强调滤波
    D_0:高通范围
    k:High_boosting系数
    output:
    new_im:处理之后的图像
    """
    L = 256
    f_min = np.min(im_ary)
    f_max = np.max(im_ary)
    M, N = 2 * im_ary.shape[0], 2 * im_ary.shape[1]
    H_1 = np.zeros((M, N, 2))  # 实部全为1，虚部全为0
    for i in range(M):
        for j in range(N):
            H_1[i, j, 0] = 1
    if Laplace == True:
        # 生成Laplace算子
        crow, ccol = int(M / 2), int(N / 2)  # 求得图像的中心点位置
        H_lap = np.zeros((M, N, 2))
        for i in range(M):
            for j in range(N):
                H_lap[i, j, 0] = -4 * (np.pi) ** 2 * ((i - crow) ** 2 + (j - ccol) ** 2)
        # 转为空间域（先对原始图像归一化）
        f_lap = Fre_filter(im_ary / f_max, H_lap)
        # 锐化之后的图像
        new_ary = im_ary - f_lap / np.max(np.abs(f_lap)) * (L - 1)

    elif High_boosting == True:
        # 生成高通高斯滤波
        sigma = D_0
        f = lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        crow, ccol = int(M / 2), int(N / 2)  # 求得图像的中心点位置
        H_lp = np.zeros((M, N, 2))
        for i in range(M):
            for j in range(N):
                H_lp[i, j, 0] = f(float(i - crow), float(j - ccol))
        H_hp = H_1 - H_lp
        # 转为空间域（先对原始图像归一化）
        g_mask = Fre_filter(im_ary / f_max, H_hp)
        # 生成High_boosting滤波
        new_ary = im_ary + k * g_mask / np.max(np.abs(g_mask)) * (L - 1)
    # 不能用简单的归一化，而是使用截断
    return truncation(new_ary, f_min, f_max)

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

def normalize(im_ary):
    """
    归一化函数
    """
    L = 256
    i_max = np.max(im_ary)
    i_min = np.min(im_ary)
    new_ary = (im_ary - i_min)/(i_max-i_min)*(L-1)
    return new_ary

if __name__ == '__main__':
    #平滑
    # 读取图像
    im = Image.open('lena.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 平滑(理想滤波器)
    new_ary1 = Smoothing(im_ary, ideal=True, Gaussian=False, D_0=60)
    plt.imshow(normalize(new_ary1), cmap='gray')
    plt.axis("off")
    plt.show()
    # 平滑(高斯滤波器)
    new_ary2 = Smoothing(im_ary, ideal=False, Gaussian=True, D_0=60)
    plt.imshow(normalize(new_ary2), cmap='gray')
    plt.axis("off")
    plt.show()
    #锐化
    # 读取图像
    im = Image.open('blurry_moon.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 锐化（Laplace算子）
    new_ary3 = Sharpening(im_ary, Laplace=True, High_boosting=False, D_0=60, k=9)
    plt.imshow(new_ary3, cmap='gray')
    plt.axis("off")
    plt.show()
    # 锐化（High boosting方法 k=1时 即unsharp masking方法）
    new_ary4 = Sharpening(im_ary, Laplace=False, High_boosting=True, D_0=60, k=1)
    plt.imshow(new_ary4, cmap='gray')
    plt.axis("off")
    plt.show()
    # 锐化（High boosting方法）
    new_ary4 = Sharpening(im_ary, Laplace=False, High_boosting=True, D_0=60, k=2)
    plt.imshow(new_ary4, cmap='gray')
    plt.axis("off")
    plt.show()