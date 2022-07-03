from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problem2a import nD_histogram

def local_eq(image,local_size):
    k = local_size
    imarry = np.array(image,dtype='int32')
    L = imarry.max()
    im = imarry
    s_data = np.zeros(im.shape)
    row_length,col_length = im.shape
    # initialize local histgram:
    hist = nD_histogram(im[0:k,0:k].reshape((k*k,1)),1,[255],[0],[255])
    for row in tqdm(range(row_length-k+1)):
        # 控制移动方向（右移）
        if row % 2 == 0:
            for col in range(col_length-k):
                if col != 0:
                    # 计算local histogram
                    sub_hist = nD_histogram(im[row:row+k,col-1].reshape((k,1)),1,[255],[0],[255])
                    add_hist = nD_histogram(im[row:row+k,col+k-1].reshape((k,1)),1,[255],[0],[255])
                    hist = hist + add_hist - sub_hist
                # local histogram equalization
                #特殊情况（边界）
                if row == 0: # 上边界
                    if col == 0: # 左上方块
                        for i in range(k//2):
                            for j in range(k//2):
                                s_data[i,j] = hist[:im[i,j]].sum()/(k*k) * L
                    elif col == col_length-k-1: # 右上方块
                        for i in range(k//2):
                            for j in range(k//2):
                                s_data[i,col+k//2+j] = hist[:im[i,col+k//2+j]].sum()/(k*k) * L
                    for i in range(k//2):
                        s_data[row+i,col+k//2] = hist[:im[row+i,col+k//2]].sum()/(k*k) * L
                if col == 0 : #左边界
                    for j in range(k//2):
                        s_data[row+k//2,col+j] = hist[:im[row+k//2,col+j]].sum()/(k*k) * L
                if col == col_length-k-1: #右边界
                    for j in range(k//2):
                        s_data[row+k//2,col+k//2+j] = hist[:im[row+k//2,col+k//2+j]].sum()/(k*k) * L
                # 一般情况取local square的中点做local histogram equalization
                s_data[row+k//2,col+k//2] = hist[:im[row+k//2,col+k//2]].sum()/(k*k) * L

            if row != row_length-k: # 控制移动方向（下移）
                sub_hist = nD_histogram(im[row,col_length-k:col_length].reshape((k,1)),1,[255],[0],[255])
                add_hist = nD_histogram(im[row+k,col_length-k:col_length].reshape((k,1)),1,[255],[0],[255])
                hist = hist + add_hist - sub_hist
        # 控制移动方向（左移）
        else:
            for col in range(col_length-1,k-1,-1):
                if col != col_length - 1:
                    # 计算local histogram
                    sub_hist = nD_histogram(im[row:row+k,col].reshape((k,1)),1,[255],[0],[255])
                    add_hist = nD_histogram(im[row:row+k,col-k+1].reshape((k, 1)), 1, [255], [0], [255])
                    hist = hist + add_hist - sub_hist

                # local histogram equalization
                #特殊情况（边界）
                if row == row_length-k: # 下边界
                    if col == k: # 左下方块
                        for i in range(k//2):
                            for j in range(k//2):
                                s_data[row+k//2+i,j] = hist[:im[row+k//2+i,j]].sum()/(k*k) * L
                    elif col == col_length-1: # 右下方块
                        for i in range(k//2):
                            for j in range(k//2):
                                s_data[row+k//2+i,col-j] = hist[:im[row+k//2+i,col-j]].sum()/(k*k) * L
                    for i in range(k//2):
                        s_data[row+k//2+i,col-k//2] = hist[:im[row+k//2+i,col-k//2]].sum()/(k*k) * L
                if col == k: # 左边界
                    for j in range(k//2):
                        s_data[row+k//2,col-k//2-j] = hist[:im[row+k//2,col-k//2-j]].sum()/(k*k) * L
                if col == col_length-1: # 右边界
                    for j in range(k//2):
                        s_data[row+k//2,col-j] = hist[:im[row+k//2,col-j]].sum()/(k*k) * L
                # 一般情况取local square的中点做local histogram equalization
                s_data[row+k//2,col-k//2] = hist[:im[row+k//2,col-k//2]].sum()/(k*k) * L

            if row != row_length-k:  # 控制移动方向（下移）
                sub_hist = nD_histogram(im[row,0:k].reshape((k,1)),1,[255],[0],[255])
                add_hist = nD_histogram(im[row+k,0:k].reshape((k, 1)),1,[255],[0],[255])
                hist = hist + add_hist - sub_hist
    return s_data

if __name__ == '__main__':
    im = Image.open('daitu.jpg').convert('L')
    s_data = local_eq(im,125)
    plt.imshow(s_data, cmap='gray')
    plt.savefig('problem3b.jpg')