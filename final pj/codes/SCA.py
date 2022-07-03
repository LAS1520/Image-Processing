import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import random
from Histgram import nD_histogram

class SCA:
    # 用正弦余弦算法解决聚类问题
    def __init__(self, image, k=4, agent_num=30, max_iters=200):
        """
        初始化
        :param image: 输入的图像
        :param k: 聚类的数目
        :param agent_num: 搜索智能体的数目
        :param max_iters: 最大迭代次数
        """
        self.image = image
        self.shape = image.shape  # 图像大小
        # 直方图
        m, n = self.shape
        self.hist = nD_histogram(self.image.reshape((m*n, 1)), 1, [256], [0], [256])
        # 为阈值设定界限
        self.low_bound = 1
        self.upper_bound = 255
        self.k = k - 1  # 分成k类需要k-1个阈值
        self.agent_num = agent_num
        self.max_iters = max_iters
        # 包含所有智能体位置的矩阵
        self.X = np.zeros((self.agent_num, self.k))
        # 存储目标函数值
        self.obj_values = [0.0] * self.agent_num
        # 初始化
        self.initialization()
        self.fitness()
        # 最优
        self.P = None

    def initialization(self):
        """
        初始化搜索智能体
        :return:
        """
        # 包含所有智能体位置(阈值)的矩阵
        X = np.random.randint(self.low_bound, self.upper_bound+1, size=(self.agent_num,self.k), dtype='int32')
        self.X = X
        return X

    def fitness(self):
        """
        求目标函数
        :param x:
        :return:
        """
        for i in range(self.agent_num):
            x = self.X[i, :]
            m, n = self.shape
            image_list = [self.image.reshape(m*n)] * self.k
            images = np.stack(image_list, axis=1)
            value = np.sum((images - x) ** 2)
            self.obj_values[i] = value

    def fitness2(self):
        """
        计算适应度函数：采用OTSU计算类间方差
        :return: f_list 不同染色体个体的适应度函数
        """
        L = 256
        m, n = self.shape
        N = m * n
        # 图片的概率直方图
        imhist = self.hist / N
        standn = np.arange(L)
        # 图片的全局像素平均值
        mu_g = np.sum(standn * self.hist) / N
        # 适应度函数 f_list
        f_list = [0.0] * self.agent_num  # pop_size 是可变的
        for i in range(self.agent_num):
            # 阈值数组
            threshold_list = list(self.X[i, :])
            threshold_list.append(256)
            # 计算类间方差
            for j in range(self.k):
                if j == 0:
                    w = np.sum(imhist[0:threshold_list[j]])
                    if w > 0:
                        mu = np.dot(imhist[0:threshold_list[j]], np.arange(0, threshold_list[j])) / w
                        f_list[i] += w * (mu - mu_g) ** 2
                else:
                    w = np.sum(imhist[threshold_list[j]:threshold_list[j + 1]]) / N
                    if w > 0:
                        mu = np.dot(imhist[threshold_list[j]:threshold_list[j + 1]],
                                    np.arange(threshold_list[j], threshold_list[j + 1])) / w
                        f_list[i] += w * (mu - mu_g) ** 2
        # print(end - start)  # 打印运行时间
        self.obj_values = f_list
        return f_list

    def search(self):
        """
        迭代寻找最优
        :return:
        """
        # 更新
        for iter in range(1,self.max_iters):
            self.P = self.X[np.argmin(self.obj_values), :]  # min
            # update r1
            a = 2
            r_1 = a - iter * a / self.max_iters
            # update r2,r3,r4
            r_2 = 2 * np.pi * np.random.rand(self.agent_num, self.k)
            r_3 = 2 * np.random.rand(self.agent_num, self.k)
            r_4 = np.random.rand(self.agent_num, self.k)
            # update positions of X
            boolmatrix = r_4 < 0.5
            add_sin = r_1 * np.sin(r_2) * np.abs(r_3 * self.P - self.X) * boolmatrix
            add_cos = r_1 * np.cos(r_2) * np.abs(r_3 * self.P - self.X) * (1 - boolmatrix)
            self.X += add_sin.astype('int32')
            self.X += add_cos.astype('int32')
            # # 边界检查（截断）
            low_judge = self.X < self.low_bound
            self.X = self.X * (1 - low_judge) + low_judge * self.low_bound
            upper_judge = self.X > self.upper_bound
            self.X = self.X * (1 - upper_judge) + upper_judge * self.upper_bound
            # 更新 obj_values
            self.fitness()

    def threshold_seg(self):
        """
        根据阈值进行图像分割
        :return: new_image 分割之后的图像 ndarray
        """
        new_image = np.zeros(self.shape, dtype='int32')
        # 选择最小目标函数
        threshold_list = list(self.X[np.argmin(self.obj_values), :])  # min
        threshold_list.sort()
        threshold_list.append(255)
        # k 阈值分类
        for i in range(self.k+1):
            if i == 0:
                judge1 = self.image >= 0
                judge2 = self.image < threshold_list[i]
                new_image += threshold_list[i] * judge1 * judge2
            else:
                judge1 = self.image >= threshold_list[i-1]
                judge2 = self.image < threshold_list[i]
                new_image += threshold_list[i] * judge1 * judge2
        return new_image



if __name__ == '__main__':
    imagesrc = cv2.imread("photo1.webp")
    image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    mySCA = SCA(image)
    mySCA.search()
    new = mySCA.threshold_seg()
    print(mySCA.P)
    plt.imshow(new, cmap='gray')
    plt.axis('off')
    plt.show()


