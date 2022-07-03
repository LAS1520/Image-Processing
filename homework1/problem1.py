from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = Image.open('daitu.jpg').convert('L')
im.save('daitu_grey.jpg')
im_ary = np.array(im,dtype = 'int32')
print(im_ary.shape,im_ary.dtype)
L = im_ary.max()
print(im_ary,im_ary.max())
#灰度值变换：y=0.5x,y=2x-0.5L,y=0.25x+3L/4;(L/3,L/6),(5L/7,13L/14)
with np.nditer(im_ary, op_flags=['readwrite']) as it:
    for x in it:
        if x < L / 3:
            x[...] = 0.5 * x
            # item = int(item * 0.5)
        elif L/3 <= x < 5*L/7:
            x[...] = 2 * x - 0.5 * L
        else:
            x[...] = 0.25 * x + 3/4 * L
print(im_ary,im_ary.max())
plt.imshow(im_ary,cmap='gray')
plt.savefig('problem1.jpg')


