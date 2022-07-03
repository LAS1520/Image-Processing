import numpy as np
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def im_erode_expand(image, target=255, k=3):
    """
    实现腐蚀或膨胀
    :param image: 输入图像
    :param target: 目标物体
    :param k: 腐蚀或膨胀尺度
    :return: 返回腐蚀或膨胀后图像
    """
    m, n = image.shape
    # padding
    edge = k//2
    row = m + edge * 2
    col = n + edge * 2
    if target == 255:
        img = np.ones((row,col),dtype='int32')*255
    else:
        img = np.zeros((row,col),dtype='int32')
    img[edge:row-edge,edge:col-edge] = image
    # kernal
    if target == 255:
        kernal = np.zeros((k, k))
    else:
        kernal = np.ones((k, k))*255
#     for i in range(k):
#         for j in range(k):
#             if abs(i - edge) + abs(j - edge) <= edge:
#                 kernal[i, j] = target
    new_img = img.copy()
    # 遍历
    for i in range(m):
        for j in range(n):
            # 检查边界
            if img[edge+i,edge+j] == target:
                if img[edge+i-1,edge+j] == target and img[edge+i+1,edge+j] == target \
                        and img[edge+i,edge+j-1] == target and img[edge+i,edge+j+1] == target:
                    e = 1
                else:  # 边界点
                    new_img[i:i+2*edge+1, j:j+2*edge+1] = kernal
    return new_img[edge:row-edge,edge:col-edge]

def im_open(image, target=255, k=3):
    inter_ary = im_erode_expand(image,target, k) # 先腐蚀
    new_target = 0 if target == 255 else 255
    new_ary = im_erode_expand(inter_ary,new_target, k) # 再膨胀
    return new_ary

def im_close(image, target=255, k=3):
    new_target = 0 if target == 255 else 255
    inter_ary = im_erode_expand(image,new_target, k) # 先膨胀
    new_ary = im_erode_expand(inter_ary,target, k) # 再腐蚀
    return new_ary

if __name__ == '__main__':
    pass
