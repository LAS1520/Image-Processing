from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def Histequal(data,L):
    eq_data = data/L
    s_data = np.zeros(eq_data.shape)
    a,b = eq_data.shape
    for i in tqdm(range(a)):
        for j in range(b):
            booldata = (eq_data <= eq_data[i][j])
            s_data[i][j] = int(booldata.astype('int32').sum()/(a * b) * L)
    return s_data

if __name__ == '__main__':
    im = Image.open('daitu.jpg').convert('L')
    im.save('daitu_grey.jpg')
    im_ary = np.array(im, dtype='int32')
    L = im_ary.max()
    eq_im = Histequal(im_ary,L)
    plt.imshow(eq_im, cmap='gray')
    plt.savefig('problem3a.jpg')

