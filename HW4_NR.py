#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from numpy import *


# Logistic函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 生成X
np.random.seed(42)
N = 1600
mu = 0
sigma1 = 1
sigma2 = 1
sigma3 = 1
X0 = np.ones([N, 1])
X1 = np.random.normal(mu, sigma1, size=(N, 1))
X2 = np.random.normal(mu, sigma2, size=(N, 1))
X3 = np.random.normal(mu, sigma3, size=(N, 1))
X = np.concatenate((X0, X1, X2, X3), axis=1)
X = np.mat(X)

# 设置真实参数β
beta = np.mat([-1, 2, -3, 0.5]).reshape(4, 1)

# 真实标签
y = sigmoid(X * beta)
e_k = []


# 牛顿法
def Newton(dataMatIn, classLabels):
    maxiter = 200  # 设置最大迭代次数
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMatrix)
    w = np.ones((n, 1))
    k = 0
    hxhx_1 = lambda x: sigmoid(x) * (sigmoid(x) - 1)  # 设置函数计算h(X)(h(X)-1)
    while k < maxiter:
        h = sigmoid(dataMatrix * w)
        error = (labelMat - h)
        lw1 = dataMatrix.transpose() * error
        t = np.array(h)
        A = np.identity(m) * hxhx_1(t)
        # 计算海森矩阵
        Hmat = dataMatrix.transpose() * A * dataMatrix
        w = w - Hmat.I * lw1
        k += 1
        e_k.append(w - beta)
    print(w)
    return w


w = Newton(X, y)
e_k = np.array(e_k).reshape(200, 4).transpose(1, 0)
print(e_k.shape)

# 画箱线图
import matplotlib.pyplot as plt
for i in range(4):
    plt.boxplot(e_k[i])
    plt.legend()
    plt.title('N={} β^_{} - β_{}'.format(N, i, i))
    plt.savefig('N{}j{}.jpg'.format(N,i))
