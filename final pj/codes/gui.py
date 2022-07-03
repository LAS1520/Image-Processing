from asyncio.windows_events import NULL
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
# PySide6-uic demo.ui -o ui_demo.py
from ui_demo import Ui_MainWindow
from kmeans import k_means
from fcm import FCM
from GA import GA_seg
from PSO import PSO
from SCA import SCA
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow() # UI类的实例化
        self.ui.setupUi(self)
        self.ori_img = None # 输入的原始图像
        self.dir = None # 输入图像的url地址
        self.is_ori = False # 是否已输入原始图像
        self.k = 2 # 图像分割类别数
        self.bind() # 绑定信号与槽

    def bind(self):
        # self.ui.___ACTION___.triggered.connect(___FUNCTION___)
        # self.ui.___BUTTON___.clicked.connect(___FUNCTION___)
        # self.ui.___COMBO_BOX___.currentIndexChanged.connect(___FUNCTION___)
        # self.ui.___SPIN_BOX___.valueChanged.connect(___FUNCTION___)
        # 自定义信号.属性名.connect(___FUNCTION___)
        self.ui.input_image.clicked.connect(self.origin_input) # 点击输入图像按钮，选择原始图像
        self.ui.output_image.clicked.connect(self.output) # 点击生成图像按钮，生成聚类后的图像

    def origin_input(self):
        dir = QMainWindow()
        file_choose = QFileDialog(dir)
        file_dir = file_choose.getOpenFileName(dir, "原始图像", "", "*.png;;*.jpg;;All Files(*)") # 输入图片地址
        self.dir = file_dir[0]
        self.ori_img = cv2.imread(self.dir, cv2.IMREAD_GRAYSCALE)
        self.ui.Origin_image.setStyleSheet(f"border-image: url({file_dir[0]});") # 输出原始图像
        self.is_ori = True
    
    def output(self):
        if self.ui.k_value.value() >= 2: # 若k>=2，则使用输入的k，否则使用缺省值k=4
            self.k = self.ui.k_value.value()
        if self.ui.Kmeans_box.isChecked():
            self.output_kmeans()
        if self.ui.FCM_box.isChecked():
            self.output_FCM()
        if self.ui.GA_box.isChecked():
            self.output_GA()
        if self.ui.PSO_box.isChecked():
            self.output_PSO()
        if self.ui.SCA_box.isChecked():
            self.output_SCA()
    
    def output_kmeans(self):
        if self.is_ori: # 若无输入图像，则无法生成图像
            Kmeans_img = Image.open(self.dir).convert('L')
            Kmeans_img = np.array(Kmeans_img, dtype='int32')
            Kmeans_img = np.expand_dims(Kmeans_img, axis=2)
            Kmeans_img = k_means(Kmeans_img, k=self.k)
            Kmeans_img = np.squeeze(Kmeans_img, axis=2)
            Kmeans_img = np.array(Kmeans_img, dtype='uint8')
            Kmeans_img = Image.fromarray(Kmeans_img)
            Kmeans_img = Kmeans_img.toqpixmap()
            Kmeans_img = Kmeans_img.scaledToHeight(self.ui.Kmeans_image.height())
            Kmeans_img = Kmeans_img.scaledToWidth(self.ui.Kmeans_image.width())
            self.ui.Kmeans_image.setPixmap(Kmeans_img)
    
    def output_FCM(self):
        if self.is_ori: # 若无输入图像，则无法生成图像
            FCM_img = FCM(self.dir,cluster_n=self.k,iter_num=20)
            FCM_img = Image.fromarray(FCM_img)
            FCM_img = FCM_img.toqpixmap()
            FCM_img = FCM_img.scaledToHeight(self.ui.FCM_image.height())
            FCM_img = FCM_img.scaledToWidth(self.ui.FCM_image.width())
            self.ui.FCM_image.setPixmap(FCM_img)

    def output_GA(self):
        if self.is_ori: # 若无输入图像，则无法生成图像
            imagesrc = cv2.imread(self.dir)
            image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
            GA_img = GA_seg(image, k=self.k)
            GA_img = GA_img.train()
            GA_img = np.array(GA_img, dtype='uint8')
            GA_img = Image.fromarray(GA_img)
            GA_img = GA_img.toqpixmap()
            GA_img = GA_img.scaledToHeight(self.ui.GA_image.height())
            GA_img = GA_img.scaledToWidth(self.ui.GA_image.width())
            self.ui.GA_image.setPixmap(GA_img)

    def output_PSO(self):
        if self.is_ori: # 若无输入图像，则无法生成图像
            imagesrc = cv2.imread(self.dir)
            image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
            PSO_img = PSO(image, k=self.k)
            PSO_img.run()
            PSO_img = PSO_img.threshold_seg()
            PSO_img = np.array(PSO_img, dtype='uint8')
            PSO_img = Image.fromarray(PSO_img)
            PSO_img = PSO_img.toqpixmap()
            PSO_img = PSO_img.scaledToHeight(self.ui.PSO_image.height())
            PSO_img = PSO_img.scaledToWidth(self.ui.PSO_image.width())
            self.ui.PSO_image.setPixmap(PSO_img)
    
    def output_SCA(self):
        if self.is_ori: # 若无输入图像，则无法生成图像
            imagesrc = cv2.imread(self.dir)
            image = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)
            SCA_img = SCA(image, k=self.k)
            SCA_img.search()
            SCA_img = SCA_img.threshold_seg()
            SCA_img = np.array(SCA_img, dtype='uint8')
            SCA_img = Image.fromarray(SCA_img)
            SCA_img = SCA_img.toqpixmap()
            SCA_img = SCA_img.scaledToHeight(self.ui.SCA_image.height())
            SCA_img = SCA_img.scaledToWidth(self.ui.SCA_image.width())
            self.ui.SCA_image.setPixmap(SCA_img)

if __name__ == '__main__':
    app = QApplication([]) # 启动一个应用
    window = MainWindow() # 实例化主窗口
    window.show() # 展示主窗口
    app.exec() # 避免程序执行到这一行后直接退出