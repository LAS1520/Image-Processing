import cv2
import numpy as np
import time

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
def Cen_Iter( data, U):
    # 初始化中心点
    U_m = U**2
    sum_u = np.sum(U_m, axis=1)
    ux = np.sum(np.dot(U_m, data), axis=1)
    center = np.true_divide(ux, sum_u)
    return center
    

# 更新隶属度矩阵
def U_Iter(data, U, c):
    cluster_n,sample_num = U.shape
    data2_trans = (data**2).reshape((1,sample_num))
    c2_trans = (c**2).reshape((cluster_n,1))
    dist1 = np.dot(np.ones((cluster_n,1)), data2_trans) + \
             np.dot(c2_trans, np.ones((1, sample_num))) - 2*np.dot(c.reshape((cluster_n,1)), data.T)
    dist2 = np.true_divide(np.ones(U.shape),dist1)
    sum_k = np.sum(dist2, axis=0)
    U = np.multiply(dist1, sum_k)
    U = np.true_divide(np.ones(U.shape), U)
    return U


def FCM(img_path,cluster_n=5,iter_num=10): # 迭代次数默认为10
    # 读入灰度图像
    start = time.time()
    img = cv2.imread(img_path,0)
    # 将图片拉成一列
    data = img.reshape(img.shape[0]*img.shape[1],1)
    data = np.array(data, dtype=np.uint32)
    print("开始聚类")
    sample_num = len(data)
    # 初始化隶属度矩阵U
    U = Initial_U(sample_num, cluster_n)
    for i in range(0, iter_num):
        C = Cen_Iter(data, U)
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
    return new_img




