import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def my_transform(im_ary, MA, target_m=1000, target_n=1000):
    """
    基于仿射变换的前向图变换
    :param im_ary: 输入图像
    :param MA: 仿射变换的变换矩阵
    :param target_m: 目标图像的行数
    :param target_n: 目标图像的列数
    :return:
    """
    trans_matrix = MA
    # 输入图像的形状
    m,n,q = im_ary.shape
    new_img = np.zeros((target_m,target_n,3),dtype = 'int32')
    sum_changed = np.zeros((target_m,target_n))
    # 计算坐标（i，j）变换之后的新坐标，并通过前向图变换给新坐标赋值
    for i in tqdm(range(m)):
        for j in range(n):
            # 输入图像的坐标向量
            vector = np.array([i,j,1]).reshape(3,1)
            # 变换后得到的目标图像的坐标向量
            newvect = np.dot(trans_matrix,vector)
            new_i = int(np.round(newvect[0]))
            new_j = int(np.round(newvect[1]))
            # 边界判断
            if new_i >= target_m or new_j >= target_n or new_i < 0 or new_j < 0:
                continue
            # 前向图赋值
            if sum_changed[new_i,new_j] == 0:
                sum_changed[new_i,new_j] += 1
                new_img[new_i,new_j,0] = int(im_ary[i,j,0])
                new_img[new_i,new_j,1] = int(im_ary[i,j,1])
                new_img[new_i,new_j,2] = int(im_ary[i,j,2])
    return new_img

def Shear(alpha=0,beta=0):
    trans_matrix1 = np.eye(3)
    trans_matrix1[1,0] = alpha
    trans_matrix2 = np.eye(3)
    trans_matrix2[0,1] = beta
    return np.dot(trans_matrix1, trans_matrix2)

def Scaling(sx=1,sy=1):
    trans_matrix = np.eye(3)
    trans_matrix[0,0] = sx
    trans_matrix[1,1] = sy
    return trans_matrix

def Translation(x=0,y=0):
    trans_matrix = np.eye(3)
    trans_matrix[0,2] = x
    trans_matrix[1,2] = y
    return trans_matrix

def Rotation(theta=0):
    trans_matrix = np.eye(3)
    trans_matrix[0,0] = np.cos(theta)
    trans_matrix[0,1] = np.sin(theta)
    trans_matrix[1,1] = np.cos(theta)
    trans_matrix[1,0] = -np.sin(theta)
    return trans_matrix

def Rigid(x=0,y=0,theta=0):
    trans_matrix1 = Translation(x,y)
    trans_matrix2 = Rotation(theta)
    trans_matrix = np.dot(trans_matrix1,trans_matrix2)
    return trans_matrix

def Affine(parameters):
    # 仿射
    t1 = Shear(alpha=parameters['shear_a'], beta=parameters['shear_b'])
    t2 = Scaling(sx=parameters['scale_a'],sy=parameters['scale_b'])
    t3 = Translation(x=parameters['trans_a'],y=parameters['trans_b'])
    t4 = Rotation(theta=parameters['theta'])
    trans_matrix = np.dot(t1,np.dot(t2,np.dot(t3,t4)))
    return trans_matrix

def get_coord(shape):
    """
    获得浮动图片的所有坐标
    :param shape: 图片形状
    :return:
    """
    # 只考虑二维图片
    m,n = shape
    seq_x = np.arange(m,dtype='int32')
    seq_y = np.arange(n,dtype='int32')
    x, y = np.meshgrid(seq_x,seq_y)
    # 坐标矩阵
    coord = np.stack((np.transpose(x),np.transpose(y)))
    return coord

def linear_interpolation(new_coord, float_img):
    """
    线性插值拟合（从浮动图像到参考图像）
    :param new_coord: 坐标矩阵
    :param float_img: 浮动图像
    :return: 参考图像
    """
    coord_floor = np.floor(new_coord).astype(int)
    q_coord = new_coord - coord_floor
    # 可以广播到三通道图像
    get_img = lambda a,b: float_img[:,np.clip(a, 0, float_img.shape[1] - 1),np.clip(b, 0, float_img.shape[2] - 1)]
    # 双线性插值
    a = (1-q_coord[0]) * (1-q_coord[1]) * get_img(coord_floor[0], coord_floor[1])
    b = (1-q_coord[0]) * q_coord[1] * get_img(coord_floor[0], coord_floor[1]+1)
    c = q_coord[0] * (1-q_coord[1]) * get_img(coord_floor[0]+1, coord_floor[1])
    d = q_coord[0] * q_coord[1] * get_img(coord_floor[0]+1, coord_floor[1]+1)
    return a+b+c+d

def Trans_coord(M, coord):
    shape = coord.shape
    MA = M[:2,:2]
    Ts = M[:2,2].reshape((2,1))
    new_coord = np.dot(MA, coord.reshape(shape[0],-1)) + Ts
    return new_coord.reshape(shape)

