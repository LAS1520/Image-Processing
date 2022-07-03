from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problem2a import nD_histogram

def local_hist(image,local_size):
    k = local_size
    im = np.array(image,dtype='int32')
    row_length,col_length = im.shape
    # 选择合适的local_size
    if row_length%k or col_length%k:
        print("Wrong local_size")
        return 0
    # initialize local histgram:
    hist = nD_histogram(im[0:k,0:k].reshape((k*k,1)),1,[50],[0],[255])
    for row in tqdm(range(row_length-k)):
        if row % 2 == 0:
            for col in range(col_length-k):
                if col == 0:
                    continue
                sub_hist = nD_histogram(im[row:row+k,col-1].reshape((k,1)),1,[50],[0],[255])
                add_hist = nD_histogram(im[row:row+k,col+k-1].reshape((k,1)),1,[50],[0],[255])
                hist = hist + add_hist - sub_hist
            if row != row_length-k-1:
                sub_hist = nD_histogram(im[row,col_length-k:col_length].reshape((k,1)),1,[50],[0],[255])
                add_hist = nD_histogram(im[row+k,col_length-k:col_length].reshape((k,1)),1,[50],[0],[255])
                hist = hist + add_hist - sub_hist
        else:
            for col in range(col_length-1,k-1,-1):
                if col == col_length - 1:
                    continue
                sub_hist = nD_histogram(im[row:row+k,col].reshape((k,1)),1,[50],[0],[255])
                add_hist = nD_histogram(im[row:row+k,col-k+1].reshape((k, 1)), 1, [50], [0], [255])
                hist = hist + add_hist - sub_hist
            if row != row_length-k-1:
                sub_hist = nD_histogram(im[row,0:k].reshape((k,1)),1,[50],[0],[255])
                add_hist = nD_histogram(im[row+k,0:k].reshape((k, 1)),1,[50],[0],[255])
                hist = hist + add_hist - sub_hist


if __name__ == '__main__':
    im = Image.open('daitu.jpg').convert('L')
    local_hist(im,125)