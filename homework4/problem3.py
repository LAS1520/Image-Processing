import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def alter_filter(shape,W,D_0,n):
    """
    生成带阻滤波器
    input:
    W:带宽
    D_0:带宽的径向中心
    n:布特沃斯滤波器的阶数
    """
    M,N = 2*shape[0],2*shape[1]
    #生成布特沃斯滤波器
    f_1 = lambda x, y: np.sqrt((x)**2 + (y)**2)
    f_2 = lambda t, n: 1/(1+np.power(t,2*n))
    crow, ccol = int(M / 2), int(N / 2) # 求得图像的中心点位置
    H = np.zeros((M, N, 2))
    for i in range(M):
        for j in range(N):
            D = f_1(i-crow,j-ccol)
            t = D*W/(D**2-D_0**2)
            H[i,j,0] = f_2(t,n)
    return H

def dai_filter(shape,d=10):
    """
    生成条形带阻滤波器
    input:
    W:带宽
    D_0:带宽的径向中心
    n:布特沃斯滤波器的阶数
    """
    M,N = 2*shape[0],2*shape[1]
    #生成条形带阻滤波器
    crow, ccol = int(M / 2), int(N / 2) # 求得图像的中心点位置
    H = np.zeros((M, N, 2))
    for i in range(M):
        for j in range(N):
            D = np.abs(i-crow)
            if 120-d<D<120+d:
                H[i,j,0] = 0
            elif 280-d<D<280+d:
                H[i,j,0] = 0
            elif 350-d<D<350+d:
                H[i,j,0] = 0
            elif 520-d<D<520+d:
                H[i,j,0] = 0
            elif 670-d<D<670+d:
                H[i,j,0] = 0
            else:
                H[i,j,0] = 1
    for i in range(M):
        for j in range(N):
            D_i = np.abs(i-crow)
            D = np.abs(j-ccol)
            if D_i>350:
                if 120-d<D<120+d:
                    H[i,j,0] = 0
                elif 280-d<D<280+d:
                    H[i,j,0] = 0
                elif 520-d<D<520+d:
                    H[i,j,0] = 0
                elif 670-d<D<670+d:
                    H[i,j,0] = 0
    return H

if __name__ == '__main__':
    # 1
    # 读取图像
    im = Image.open('photo3.PNG').convert('L')
    im_ary = np.array(im, dtype='float32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 填充图像（padding）
    M, N = im_ary.shape
    new_ary = np.zeros((2 * M, 2 * N))
    # 计算平移之后的DFT
    for i in range(M):
        for j in range(N):
            new_ary[i, j] = im_ary[i, j] * (-1) ** (i + j)  # 平移
    F = cv2.dft(new_ary, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 画出频谱图
    f_dft = np.log(cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1)
    plt.imshow(f_dft, cmap='gray')
    plt.axis("off")
    plt.show()

    H_1 = dai_filter((M, N))
    G_1 = H_1 * F
    H_2 = alter_filter((M, N), 300, 400, 4)
    G = H_2 * G_1

    g_dft = np.log(cv2.magnitude(G[:, :, 0], G[:, :, 1]) + 1)
    plt.imshow(g_dft, cmap='gray')
    plt.axis("off")
    plt.show()

    # 第四步，计算傅里叶逆变换IDFT
    i_dft = cv2.idft(G)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    plt.imshow(result_ary[:M, :N], cmap='gray')
    plt.axis("off")
    plt.show()
    # 2
    # 读取图像
    im = Image.open('photo4.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 填充图像（padding）
    M, N = im_ary.shape
    new_ary = np.zeros((2 * M, 2 * N))
    # 计算平移之后的DFT
    for i in range(M):
        for j in range(N):
            new_ary[i, j] = im_ary[i, j] * (-1) ** (i + j)  # 平移
    F = cv2.dft(new_ary, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 增加噪声
    F[M - 105:M - 100, N - 105:N - 100] = [F.max(), 0]
    F[M + 101:M + 106, N + 101:N + 106] = [F.max(), 0]
    # 画出频谱图
    f_dft = np.log(cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1)
    plt.imshow(f_dft, cmap='gray')
    plt.axis("off")
    plt.show()
    # 计算傅里叶逆变换IDFT
    i_dft = cv2.idft(F)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    plt.imshow(result_ary[:M, :N], cmap='gray')
    plt.axis("off")
    plt.show()
    H = alter_filter((M, N), 30, 145, 4)
    G = H * F
    g_dft = np.log(cv2.magnitude(G[:, :, 0], G[:, :, 1]) + 1)
    plt.imshow(g_dft, cmap='gray')
    plt.axis("off")
    plt.show()

    # 计算傅里叶逆变换IDFT
    i_dft = cv2.idft(G)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    plt.imshow(result_ary[:M, :N], cmap='gray')
    plt.axis("off")
    plt.show()
    # 3
    # 读取图像
    im = Image.open('photo4.tif').convert('L')
    im_ary = np.array(im, dtype='float32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    # 填充图像（padding）
    M, N = im_ary.shape
    new_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            im_ary[i, j] = im_ary[i, j] + 20 * np.sin(20 * i) + 20 * np.sin(20 * j)  # 增加噪声
    # 计算平移之后的DFT
    for i in range(M):
        for j in range(N):
            new_ary[i, j] = im_ary[i, j] * (-1) ** (i + j)  # 平移
    F = cv2.dft(new_ary, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 画出频谱图
    f_dft = np.log(cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1)
    plt.imshow(f_dft, cmap='gray')
    plt.axis("off")
    plt.show()
    # 计算傅里叶逆变换IDFT
    i_dft = cv2.idft(F)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    plt.imshow(result_ary[:M, :N], cmap='gray')
    plt.axis("off")
    plt.show()
    H = alter_filter((M, N), 40, 210, 4)
    G = H * F
    g_dft = np.log(cv2.magnitude(G[:, :, 0], G[:, :, 1]) + 1)
    plt.imshow(g_dft, cmap='gray')
    plt.axis("off")
    plt.show()

    # 计算傅里叶逆变换IDFT
    i_dft = cv2.idft(G)[:, :, 0]  # 舍去虚部取实部
    result_ary = np.zeros((2 * M, 2 * N))
    for i in range(M):
        for j in range(N):
            result_ary[i, j] = i_dft[i, j] * (-1) ** (i + j)  # 平移
    plt.imshow(result_ary[:M, :N], cmap='gray')
    plt.axis("off")
    plt.show()