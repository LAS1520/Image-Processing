import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

m=2

# 初始化隶属度矩阵U
def Initial_U(sample_num, cluster_n):
    # sample_num为样本个数, cluster_n为分类数
    U = np.random.rand(cluster_n,sample_num)
    # 对 U 按列求和，然后取倒数
    col_sum = np.sum(U, axis=0)
    col_sum = 1 / col_sum
    # 确保 U 的每列和为1
    U = np.multiply(U, col_sum)
    return U

# 计算类中心
def Cen_Iter( data, U, cluster_n):
    # 初始化中心点
    center = np.zeros(cluster_n)
    for i in range(0, cluster_n):
        # 根据迭代公式进行计算
        u_ij_m = U[i, :] ** m
        sum_u = np.sum(u_ij_m)
        # 矩阵乘法
        ux = np.dot(u_ij_m, data)
        center[i] =  ux / sum_u
    return center

# 更新隶属度矩阵
def U_Iter(data,U, c):
    cluster_n,sample_num = U.shape
    for i in range(0, cluster_n):
        for j in range(0,sample_num):
            sum = 0
            # 根据隶属度矩阵迭代公式计算
            for k in range(0, cluster_n):
                temp = (np.linalg.norm(data[j] - c[i]) /
                        np.linalg.norm(data[j] - c[k])) ** (2 / (m - 1))
                sum = temp + sum
            U[i, j] = 1 / sum
    return U



def FCM(img_path,cluster_n=5,iter_num=10): # 迭代次数默认为10
    # 读入灰度图像
    start = time.time()
    img=cv2.imread(img_path,0)
    # 将图片拉成一列
    data=img.reshape(img.shape[0]*img.shape[1],1)
    print("开始聚类")
    sample_num = len(data)
    # 初始化隶属度矩阵U
    U = Initial_U(sample_num, cluster_n)
    for i in range(0, iter_num):
        C = Cen_Iter(data, U, cluster_n)
        U = U_Iter(data,U, C)
        print("第%d次迭代" % (i + 1), end="")
        print("聚类中心", C)
    # 分类标签
    label = np.argmax(U, axis=0)
    # 最后的类中心矩阵
    center = C
    print("聚类完成，开始生成图片")
    # 根据聚类结果和聚类中心构建新图像
    new_img=center[label]
    # 矩阵转成原来图片的形状
    new_img=np.reshape(new_img,img.shape)
    # 要变成图像得数据得转换成uint8
    new_img=new_img.astype('uint8')
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title("原图")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(new_img, cmap="gray")
    plt.title("FCM,%d个聚类中心"%cluster_n)
    plt.axis('off')
    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))
    plt.show()
    plt.imshow(new_img, cmap="gray")
    plt.axis('off')
    plt.savefig('FCM_Baboon')

FCM("photo2.png",cluster_n=4)




