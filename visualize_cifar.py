#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 17:01
# @Author  : Iceyhuang
# @license : Copyright(C), Tencent
# @Contact : iceyhuang@tencent.com
# @File    : visualize_cifar.py
# @Software: PyCharm
# @Version : Python 3.7.3


# 用于将cifar10的数据可视化
import os
import pickle
import numpy as np
from scipy.misc import imsave
import matplotlib.image as plimg
from PIL import Image


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        #        datadict = pickle.load(f)
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)


def visualize1():
    num = 5
    load_CIFAR_Labels("CIFAR/train/batches.meta")
    imgX, imgY = load_CIFAR_batch("CIFAR10/data_batch_{}".format(num))
    print(imgX.shape)
    print("正在保存图片:")
    #    for i in range(imgX.shape[0]):
    for i in range(10):  # 值输出10张图片，用来做演示
        #        imgs = imgX[i - 1]#?
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)  # 从数据，生成image对象
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = "img" + str(i) + '.png'
        img.save("./cifar10_images/train" + name, "png")  # 文件夹下是RGB融合后的图像
        for j in range(0, imgs.shape[0]):
            #                img = imgs[j - 1]
            img = imgs[j]
            J = j
            name = "img" + str(i) + str(J) + ".png"
            print("正在保存图片" + name)
            save_path = "./cifar10_images/train/{}/".format(num)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plimg.imsave(save_path + name, img)  # 文件夹下是RGB分离的图像

    print("保存完毕.")


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def load_train():
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    save_path = 'cifar10'
    train_path = os.path.join(save_path, 'train')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    for j in range(1, 6):
        dataName = "data_batch_" + str(j)
        path = os.path.join('CIFAR10', dataName)
        Xtr = unpickle(path)
        print(dataName + " is loading...")

        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = train_path + '/' + str(Xtr['labels'][i]) + '_' + str(
                i + (j - 1) * 10000) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
            imsave(picName, img)
        print(dataName + " loaded.")


def load_test():
    save_path = 'cifar10'
    print("test_batch is loading...")

    # 生成测试集图片
    test_path = os.path.join(save_path, 'test')
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    path = os.path.join('CIFAR10', "test_batch")
    testXtr = unpickle(path)
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = test_path + '/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
        imsave(picName, img)
    print("test_batch loaded.")


def visualize2():
    load_train()
    load_test()


if __name__ == "__main__":
    # visualize1()

    # CIFAR-10 dataset 的下载与使用、转图片https://blog.csdn.net/ctwy291314/article/details/83864405
    # data = load_file('CIFAR10/test_batch')
    # print(data.keys())
    visualize2()
