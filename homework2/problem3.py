from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def linearinter(imary,N):
    """
    input:
    imary: ndarry 输入图像矩阵
    N: int(>1) 分辨率放大倍数(N=2指放大两倍）
    output:
    newary: ndarry 放大后的图像
    """
    m,n = imary.shape
    new_m = N * (m-1) + 1
    new_n = N * (n-1) + 1
    newary = np.zeros((new_m,new_n))
    for p_row in tqdm(range(m-1)):
        for p_col in range(n-1):
            I_p0 = imary[p_row,p_col]
            I_p1 = imary[p_row,p_col+1]
            I_p2 = imary[p_row+1, p_col]
            I_p3 = imary[p_row+1, p_col+1]
            newary[p_row*N,p_col*N] = I_p0
            newary[p_row*N,p_col*N+N] = I_p1
            newary[p_row*N+N,p_col*N] = I_p2
            newary[p_row*N+N,p_col*N+N] = I_p3
            for i in range(0,N+1):
                for j in range(0,N+1):
                    s = i/N
                    r = j/N
                    newary[p_row*N+i,p_col*N+j] = (1-s)*(1-r)*I_p0 + r*(1-s)*I_p1+ \
                                                  s*(1-r)*I_p2 + s*r*I_p3
    return newary

if __name__ == "__main__":
    im = Image.open('polymersomes.tif').convert('L')
    im_ary = np.array(im, dtype='int32')
    new_image = plt.imshow(im_ary, cmap='gray')
    plt.axis("off")
    plt.show()
    new_ary = linearinter(im_ary,2)
    plt.imshow(new_ary, cmap='gray')
    plt.axis("off")
    plt.savefig('problem3.tif')