from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problem1 import BGT

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
    max_y = float('inf')
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
        if y >= max_y:
            max_y = y
            index = i
    return index

#局部阈值化处理
def local_Thresholding(image, local_size, L = 256, s_OTSU = False):
    k = local_size
    im_o = np.array(image,dtype='int32')
    row_length,col_length = im_o.shape
    #padding
    row_length += k // 2 * 2
    col_length += k // 2 * 2
    im = np.zeros((row_length,col_length),dtype='int32')
    im[k//2:row_length-k//2,k//2:col_length-k//2] = im_o
    # initialize local histgram:
    hist = nD_histogram(im[0:k,0:k].reshape((k*k,1)),1,[L],[0],[L])
    #store the result
    s_data = np.zeros(im.shape)
    for row in tqdm(range(row_length-k+1)):
        if row % 2 == 0:
            for col in range(col_length-k+1):
                if col != 0:
                    sub_hist = nD_histogram(im[row:row+k,col-1].reshape((k,1)),1,[L],[0],[L])
                    add_hist = nD_histogram(im[row:row+k,col+k-1].reshape((k,1)),1,[L],[0],[L])
                    hist = hist + add_hist - sub_hist
                if s_OTSU:
                    T = OTSU2(hist)
                    # print(T)
                    if im[row + k // 2, col + k // 2] <= T:
                        s_data[row + k // 2, col + k // 2] = 0
                    else:
                        s_data[row + k // 2, col + k // 2] = L - 1
            if row != row_length-k:
                sub_hist = nD_histogram(im[row,col_length-k:col_length].reshape((k,1)),1,[L],[0],[L])
                add_hist = nD_histogram(im[row+k,col_length-k:col_length].reshape((k,1)),1,[L],[0],[L])
                hist = hist + add_hist - sub_hist
        else:
            for col in range(col_length-1,k-2,-1):
                if col != col_length - 1:
                    sub_hist = nD_histogram(im[row:row+k,col+1].reshape((k,1)),1,[L],[0],[L])
                    add_hist = nD_histogram(im[row:row+k,col-k+1].reshape((k, 1)), 1, [L], [0], [L])
                    hist = hist + add_hist - sub_hist
                if s_OTSU:
                    T = OTSU2(hist)
                    # print(T)
                    if im[row + k // 2, col - k // 2] <= T:
                        s_data[row + k // 2, col - k // 2] = 0
                    else:
                        s_data[row + k // 2, col - k // 2] = L - 1
            if row != row_length-k:
                sub_hist = nD_histogram(im[row,0:k].reshape((k,1)),1,[L],[0],[L])
                add_hist = nD_histogram(im[row+k,0:k].reshape((k, 1)),1,[L],[0],[L])
                hist = hist + add_hist - sub_hist
    return s_data[k//2:row_length-k//2,k//2:col_length-k//2]

def local_Thresholding2(imary, local_size, L = 256, s_OTSU = False, s_Entropy = False):
    k = local_size
    im_o = imary
    row_length,col_length = im_o.shape
    #padding
    col_length += k - 1
    im = np.zeros((row_length,col_length),dtype='int32')
    im[:,k//2:col_length-k//2] = im_o
    # initialize local histgram:
    #store the result
    s_data = np.zeros(im.shape)
    for row in tqdm(range(row_length)):
        hist = nD_histogram(im[0, 0:k].reshape((k, 1)), 1, [L], [0], [L])
        for col in range(col_length-k+1):
            if col != 0:
                hist[im[row,col-1]] -= 1
                hist[im[row, col + k - 1]] += 1
            if s_OTSU:
                T = OTSU2(hist)
                # print(T)
                if im[row, col + k // 2] <= T:
                    s_data[row, col + k // 2] = 0
                else:
                    s_data[row, col + k // 2] = L - 1
    return s_data[:,k//2:col_length-k//2]

if __name__ == '__main__':
    im2 = Image.open('sine_shaded.tif').convert('L')
    im_ary2 = np.array(im2, dtype='int32')
    new_image2 = plt.imshow(im_ary2, cmap='gray')
    plt.axis("off")
    plt.show()
    new_imarry2 = local_Thresholding2(im_ary2, 15, s_OTSU=True)
    plt.imshow(new_imarry2, cmap='gray')
    plt.axis("off")
    plt.show()