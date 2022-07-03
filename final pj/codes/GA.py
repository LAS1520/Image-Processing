import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from Histgram import nD_histogram

def translate(two_num):
    """
    将二进制数转换为十进制
    :param two_num: 输入二进制数 ndarray
    :return: 十进制数 int
    """
    n = two_num.shape[0]
    ten_num = np.dot(two_num,2**np.arange(n)[::-1])
    return ten_num

class GA_seg:
    # 用遗传算法解决OTSU多阈值算法
    def __init__(self, image, k=4, pop_size=30, p_mutation=0.3, p_cross=0.8, max_iters=200):
        """
        初始化遗传算法图像分割类
        :param image: 输入图像 ndarray
        :param k: 分类的数目 int
        :param pop_size: 种群大小 int
        :param p_mutation: 染色体发生变异的概率 float
        :param p_cross: 染色体发生交叉的概率 flaot
        :param max_iters: 最大迭代次数 int
        """
        self.image = image
        self.shape = image.shape  # 图像大小
        # 直方图
        m,n = self.shape
        self.hist = nD_histogram(self.image.reshape((m*n, 1)), 1, [256], [0], [256])
        self.k = k-1  # 分成k类需要k-1个阈值
        self.pop_size = pop_size
        self.gene_length = 8  # 染色体中每个基因的长度，8bit二进制数可以表示[0,255]
        self.populations = self.initial_encode()
        self.p_mutation = p_mutation
        self.p_cross = p_cross
        self.fitvalue = np.array([1e-3] * self.pop_size)
        self.max_iters = max_iters

    def initial_encode(self):
        """
        初始随机编码出染色体
        :return: chromosomes 所有编码出的染色体个体 ndarray(size=self.pop_size*self.k*8)
        """
        m,n = self.shape
        # 图像中的所有坐标
        x_coords = np.arange(m)
        y_coords = np.arange(n)
        # # 初始化时随机选取self.pop_size*self.k个坐标
        rand_chos_x = np.random.choice(x_coords, size=self.pop_size * self.k, replace=True)  # 初始化时随机选取的坐标
        rand_chos_y = np.random.choice(y_coords, size=self.pop_size * self.k, replace=True)
        # 存储
        chromosomes = []
        for i in range(self.pop_size):
            # 编码 将所有阈值用二进制表示成基因，所有基因构成染色体
            ind_code = ''
            for j in range(self.k):
                code_str = bin(self.image[rand_chos_x[i*self.k+j], rand_chos_y[i*self.k+j]])[2:]
                if len(code_str) < 8:
                    code_str = '0' * (8 - len(code_str)) + code_str
                ind_code += code_str
                # ind_code += self.image[rand_chos_x[i*self.k+j], rand_chos_y[i*self.k+j]] << 8 * j
            # code_str = bin(ind_code)[2:]
            # 统一编码得到长度为8*self.k的基因
            assert len(ind_code) == 8 * self.k  # 检验是否编码成功
            # print(code_str)
            code = np.array(list(map(int,list(ind_code))))
            chromosomes.append(code)
        return np.array(chromosomes)

    def decode(self, chromosome):
        """
        将染色体解码为多阈值数组
        :param chromosome: 包含基因的染色体数组 ndarray
        :return: threshold_list 所有染色体对应的阈值 list
        """
        threshold_list = []
        for i in range(self.k):
            thres = translate(chromosome[i*8:(i+1)*8])
            threshold_list.append(thres)
        return threshold_list

    def fitness(self):
        """
        计算适应度函数：采用OTSU计算类间方差
        :return: f_list 不同染色体个体的适应度函数
        """
        start = time.time()
        L = 256
        m, n = self.shape
        N = m * n
        # 图片的概率直方图
        imhist = self.hist/N
        standn = np.arange(L)
        # 图片的全局像素平均值
        mu_g = np.sum(standn * self.hist) / N
        # 适应度函数 f_list
        f_list = [0.0] * self.pop_size  # pop_size 是可变的
        for i in range(self.pop_size):
            # 每一个染色体
            chromosome = self.populations[i, :]
            # 解码为阈值数组
            threshold_list = self.decode(chromosome)
            threshold_list.append(256)
            # 计算类间方差
            for j in range(self.k):
                if j == 0:
                    w = np.sum(imhist[0:threshold_list[j]])
                    if w > 0:
                        mu = np.dot(imhist[0:threshold_list[j]], np.arange(0,threshold_list[j])) / w
                        f_list[i] += w * (mu - mu_g) ** 2
                else:
                    w = np.sum(imhist[threshold_list[j]:threshold_list[j+1]])/N
                    if w > 0:
                        mu = np.dot(imhist[threshold_list[j]:threshold_list[j+1]],
                                            np.arange(threshold_list[j],threshold_list[j+1])) / w
                        f_list[i] += w * (mu - mu_g) ** 2
        end = time.time()
        # print(end - start)  # 打印运行时间
        self.fitvalue = np.array(f_list)
        return f_list

    # 适应度函数 Ostu全局算法
    def fitness2(self):
        start = time.time()
        L = 256
        m, n = self.shape
        N = m * n
        # 图片的概率直方图
        imhist = self.hist/N
        standn = np.arange(L)
        # 图片的全局像素平均值
        mu_g = np.sum(standn * self.hist) / N
        # 适应度函数 f_list
        Var = [0.0] * self.pop_size  # pop_size 是可变的
        for i in range(self.pop_size):
            # 每一个染色体
            chromosome = self.populations[i, :]
            # 解码为阈值数组
            th = self.decode(chromosome)
            th.append(255)
            th.append(0)
            th[1:] = th[:-1]
            th[0] = 0
            # 计算类间方差
            for j in range(len(th) - 1):
                w = [0.0] * (len(th) - 1)
                muT = [0.0] * (len(th) - 1)
                mu = [0.0] * (len(th) - 1)
                for k in range(th[j], th[j + 1]):
                    w[j] = w[j] + imhist[k]
                    muT[j] = muT[j] + imhist[k] * k
                if w[j] > 0:
                    mu[j] = muT[j] / w[j]
                    Var[i] = Var[i] + w[j] * pow(mu[j] - mu_g, 2)
        end = time.time()
        # print(end - start)  # 打印运行时间
        self.fitvalue = np.array(Var)

    def mutation(self):
        """
        染色体的变异
        """
        for i in range(self.pop_size):
            if np.random.rand() < self.p_mutation:  # 以p_mutation的概率进行变异
                mutate_point = np.random.randint(0, self.k*8)  # 随机产生一个实数，代表要变异基因的位置
                self.populations[i, mutate_point] ^= 1  # 将变异点的二进制为反转

    def crossover(self):
        """
        染色体的交叉
        """
        for i in range(self.pop_size):
            father = self.populations[i, :].copy()  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因
            if np.random.rand() < self.p_cross:  # 以p_cross的概率进行变异
                mother = self.populations[np.random.randint(self.pop_size), :].copy()  # 随机选择另一个体，并将该个体作为母亲
                cross_points = np.random.randint(0, self.k*8)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.populations[i, :] = child  # 替代父亲
            # new_populations = np.zeros((self.pop_size+1, self.k*8), dtype='int32')
            # new_populations[:self.pop_size, :] = self.populations
            # new_populations[self.pop_size, :] = child
            # self.populations = new_populations  # 遗传
            # self.pop_size += 1

    def selection(self):
        """
        物竞天择，淘汰选择个体
        """
        # 根据适应度函数的大小随机淘汰个体
        # idx = np.random.choice(np.arange(self.pop_size), size=int(self.pop_size*4/5), replace=True,
        #                        p=(self.fitvalue) / (self.fitvalue.sum()))
        idx = np.random.choice(np.arange(self.pop_size), size=int(self.pop_size), replace=True,
                               p=(self.fitvalue) / (self.fitvalue.sum()))
        self.populations = self.populations[idx]
        self.pop_size = self.populations.shape[0]


    def threshold_seg(self):
        """
        根据阈值进行图像分割
        :return: new_image 分割之后的图像 ndarray
        """
        new_image = np.zeros(self.shape, dtype='int32')
        # 选择最大适应度函数
        max_thres = np.argmax(self.fitvalue)
        threshold_list = self.decode(self.populations[max_thres,:])
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

    def train(self):
        # 初始化种群
        self.initial_encode()
        # 初始化适应度函数
        self.fitness()
        for iter in tqdm(range(self.max_iters)):
            # 选择
            self.selection()
            # 交叉
            self.crossover()
            # 变异
            self.mutation()
            # 求适应度函数
            self.fitness()
        return self.threshold_seg()




if __name__ == '__main__':
    imagesrc = cv2.imread("photo3.png")
    # imagesrc = cv2.imread("photo1.webp")
    image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    myGA = GA_seg(image)
    new_ary = myGA.train()
    max_thres = np.argmax(myGA.fitvalue)
    b=myGA.decode(myGA.populations[max_thres, :])
    b.sort()
    print(b)
    # print(new_ary)
    plt.imshow(new_ary, cmap='gray')
    plt.axis('off')
    plt.savefig('GA_pepper3')
    # plt.savefig('GA4_Lena')
    # GA2为修改otsu函数的计算方式
    # GA3为修改交叉为遗传而不是替代,迭代次数改为20