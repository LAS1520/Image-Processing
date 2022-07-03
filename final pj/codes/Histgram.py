import numpy as np

def nD_histogram(data,dimension,nbins,pInMin,pInMax):
    """
    计算N-dimension数据的直方图
    :param data: 输入数据 ndarray (nums,N) nums为数据个数 N为维度
    :param dimension: 维度 int
    :param nbins: 各维度的格数 list
    :param pInMin: 输入数据各维度的最小值 list
    :param pInMax: 输入数据各维度的最大值 list
    :return:
    """
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