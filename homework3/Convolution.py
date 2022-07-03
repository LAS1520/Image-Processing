import numpy as np
def convolution2d(image, kernel):
    """
    :param image: 输入图像 size更大
    :param kernel: 卷积核
    :return:
    """
    m, n = kernel.shape
    if m!=n:
        print("Wrong kernel shape!")
        return 0
    x, y = image.shape
    x = x - m + 1
    y = y - m + 1
    new_image = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            # new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)   #相关
            new_image[i,j] = np.sum(image[i:i+m, j:j+m] * kernel[m-1::-1,m-1::-1])  #卷积
    return new_image


if __name__ == '__main__':
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    b = np.zeros((15,15))
    b[7,7] = 1
    m = convolution2d(b,a)
    print(m)
