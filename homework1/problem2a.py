from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    im1 = Image.open('photo1.jpg').convert('L')
    im2 = Image.open('photo2.jpg').convert('L')
    im_ary1 = np.array(im1, dtype='int32')
    im_ary2 = np.array(im2, dtype='int32')
    a,b = im_ary1.shape
    c,d = im_ary1.shape
    arry1 = im_ary1.reshape((a * b,1))
    arry2 = im_ary2.reshape((c * d,1))
    mydata = np.hstack((arry1,arry2))
    myHistgram = nD_histogram(mydata,2,[100,100],[0,0],[255,255])
    H, xedges, yedges= np.histogram2d(im_ary1.reshape(a*b),im_ary2.reshape(c*d), bins=100)
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131, title='imshow: square bins')
    new_im = plt.imshow(H)
    plt.savefig('problem2.jpg')



