import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import random
from Histgram import nD_histogram

class PSO:
    def __init__(self, image, k=4, pop_size=30, max_iters=200, w=0.72, c1=1.49, c2=1.49):
        """
        初始化
        :param image: 输入图像 ndarray
        :param k: 分类的数目 int
        :param pop_size: 种群的大小 int
        :param max_iters: 最大迭代次数 int
        :param w: 惯性权重因子 float
        :param c1: 加速因子 float
        :param c2: 加速因子 float
        """
        self.image = image
        self.shape = image.shape  # 图像大小
        # 直方图
        m, n = self.shape
        self.hist = nD_histogram(self.image.reshape((m * n, 1)), 1, [256], [0], [256])
        self.k = k-1  # 分成k类需要k-1个阈值
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.w = w
        self.c_1 = c1
        self.c_2 = c2
        # 为粒子位置、速度设定界限
        self.X_lower = 0
        self.X_upper = 255
        self.v_lower = -4.0
        self.v_upper = 4.0
        # 包含所有粒子位置（阈值）的矩阵
        self.X = np.zeros((self.pop_size, self.k), dtype='int32')
        # 包含所有粒子速度的矩阵
        self.v = np.zeros((self.pop_size, self.k))
        # 存储目标函数值
        self.fitvalue = np.array([0.0] * self.pop_size)
        self.initialization()
        # 存储全局最优和局部最优
        self.pbest = self.X
        self.pbestvalues = np.zeros(self.pop_size)
        self.gbest = self.X[0, :]
        self.gbestvalue = 0

    def initialization(self):
        """
        随机初始化粒子群的位置和速度
        """
        self.X = np.random.randint(self.X_lower, self.X_upper+1, size=(self.pop_size,self.k), dtype='int32')
        self.v = np.random.randint(self.v_lower, self.v_upper+1, size=(self.pop_size,self.k))
        self.fitness()
        self.get_initbest()


    def fitness(self):
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
        f_list = [0.0] * self.pop_size  # pop_size 是可变的
        for i in range(self.pop_size):
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
        self.fitvalue = np.array(f_list)
        return f_list

    def get_initbest(self):
        """
        获得局部和全局最优位置
        """
        self.gbest = self.X[self.fitvalue.argmax(), :].copy()
        self.gbestvalue = self.fitvalue.max()
        self.pbest = self.X.copy()
        self.pbestvalues = self.fitvalue.copy()

    def run(self):
        """
        迭代优化
        """
        # 迭代优化
        for i in range(self.max_iters):
            # 速度更新
            self.v = self.w * self.v + \
                     self.c_1 * np.random.rand(self.pop_size,self.k) * (self.pbest - self.X) + \
                self.c_2 * np.random.rand(self.pop_size,self.k) * (self.gbest - self.X)
            # 防止越界 截断处理
            lower_judge = self.v < self.v_lower
            self.v = self.v * (1 - lower_judge) + lower_judge * self.v_lower
            upper_judge = self.v > self.v_upper
            self.v = self.v * (1 - upper_judge) + upper_judge * self.v_upper
            # 粒子位置更新
            self.X += self.v.astype('int32')
            # 防止越界 截断处理
            lower_judge = self.X < self.X_lower
            self.X = self.X * (1 - lower_judge) + lower_judge * self.X_lower
            upper_judge = self.X > self.X_upper
            self.X = self.X * (1 - upper_judge) + upper_judge * self.X_upper
            # 适应度函数更新
            self.fitness()
            # pbest,gbest 更新
            for j in range(self.pop_size):
                if self.fitvalue[j] > self.pbestvalues[j]:
                    self.pbestvalues[j] = self.fitvalue[j]
                    self.pbest[j, :] = self.X[j, :].copy()
            if self.gbestvalue < self.pbestvalues.max():
                self.gbestvalue = self.pbestvalues.max()
                self.gbest = self.X[self.pbestvalues.argmax()].copy()

        return self.gbest, self.gbestvalue

    def threshold_seg(self):
        """
        根据阈值进行图像分割
        :return: new_image 分割之后的图像 ndarray
        """
        new_image = np.zeros(self.shape, dtype='int32')
        # 选择最小目标函数
        threshold_list = list(self.gbest)
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
    imagesrc = cv2.imread("photo3.png")
    image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    myPSO = PSO(image)
    myPSO.run()
    print(myPSO.gbest)
    new = myPSO.threshold_seg()
    plt.imshow(new, cmap='gray')
    plt.axis('off')
    plt.savefig('PSO_pepper')