from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def filter2d(image, kernel):
    """
    计算图像和滤波器的卷积
    :param image: 输入图像 size更大
    :param kernel: 滤波器（卷积核）
    :return: 卷积后的图像
    """
    m, n = kernel.shape
    if m!=n:
        print("Wrong kernel shape!")
        return 0
    x, y = image.shape
    # padding
    pad_x = x + m - 1
    pad_y = y + m - 1
    pad_image = np.zeros((pad_x,pad_y))
    pad_image[m//2:pad_x-m//2,m//2:pad_y-m//2] = image
    # store result
    new_image = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            # new_image[i][j] = np.sum(pad_image[i:i+m, j:j+m]*kernel)   #相关
            new_image[i,j] = np.sum(pad_image[i:i+m, j:j+m] * kernel[m-1::-1,m-1::-1])  #卷积
    return new_image
def Gaussian2d(k_size,sigma):
    """
    生成高斯滤波器
    :param k_size: 滤波器大小,奇数
    :param sigma: 高斯分布的方差
    :return: 滤波器
    """
    f = lambda x, y: np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = np.zeros((k_size,k_size))
    trans_size = int((k_size-1)/2)
    for i in range(k_size):
        for j in range(k_size):
            kernel[i,j] = f(i-trans_size,j-trans_size)
    return kernel/np.sum(kernel)

def Smoothing(imary, k_size, Average=False, Gaussian=False, sigma=1):
    """
    平滑操作
    :param image: 输入图像
    :param k_size: 卷积核大小
    :param Average: 均值滤波
    :param Gaussian: 高斯滤波
    :param sigma: 高斯滤波的标准差
    :return: 平滑操作后的输出图像
    """
    if Average:
        kernel = np.ones((k_size,k_size))/k_size**2
        newary = filter2d(imary,kernel)
        return newary
    elif Gaussian:
        kernel = Gaussian2d(k_size,sigma)
        newary = filter2d(imary,kernel)
        return newary
def Sharpening(imary, Laplace=False, Unsharpmask=False, Highboost=False, k=1):
    """
    图像锐化处理
    :param imary: 输入图像
    :param Laplace: 是否采用Laplace
    :param Unsharpmask: 是否采用 Unsharp masking
    :param Highboost: 是否采用 Highboost filtering
    :param k: Highboost filtering 的倍数
    :return: 锐化后的图像
    """
    # 生成laplace卷积核
    if Laplace:
        kernel = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                if abs(i-1) + abs(j-1) == 1:
                    kernel[i,j] = 1
                elif abs(i-1) + abs(j-1) == 0:
                    kernel[i,j] = -4
        g_ary = filter2d(imary,kernel)
        newary = imary - g_ary
    elif Unsharpmask == True:
        smoothary = Smoothing(imary, 3, Average=True)
        g_mask = imary - smoothary
        newary = imary + g_mask
    elif Highboost == True:
        smoothary = Smoothing(imary, 3, Average=True)
        g_mask = imary - smoothary
        newary = imary + k * g_mask
    return newary

if __name__ == '__main__':
    # 1 Smoothing
    # im = Image.open('test_pattern.tif').convert('L')
    # im_ary = np.array(im, dtype='int32')
    # plt.imshow(im_ary, cmap='gray')
    # plt.axis("off")
    # plt.show()
    # new_imary1 = Smoothing(im_ary, 9, Average=True)
    # plt.imshow(new_imary1, cmap='gray')
    # plt.axis("off")
    # plt.show()
    # new_imary2= Smoothing(im_ary, 13, Gaussian=True, sigma=2)
    # plt.imshow(new_imary2, cmap='gray')
    # plt.axis("off")
    # plt.show()
    # 2 Sharpening
    im = Image.open('blurry_moon.tif').convert('L')
    im_ary = np.array(im, dtype='int32')
    plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    new_imary1 = Sharpening(im_ary, Laplace=True)
    plt.imshow(new_imary1, cmap='gray')
    plt.axis("off")
    plt.savefig('Laplace_sharpen.tif')
    new_imary2= Sharpening(im_ary,  Unsharpmask=True)
    plt.imshow(new_imary2, cmap='gray')
    plt.axis("off")
    plt.savefig('Unsharp_mask_sharpen.tif')
    new_imary2= Sharpening(im_ary,  Highboost=True, k=2)
    plt.imshow(new_imary2, cmap='gray')
    plt.axis("off")
    plt.savefig('Highboost_sharpen.tif')