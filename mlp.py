import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
from numpy import *

np.random.seed(40)
x_data = np.random.uniform(100, size=[100])
y_data = 3 * x_data + 2 + np.random.uniform(0, 30, size=[100])

# 直接使用x_data和y_data创建TensorDataset对象
train_ids = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
#学习率
lr = 0.0005
#截距
b = 2.0
#斜率
k = 3
#最大迭代次数
epochs = 1000

# 使用均方误差公式计算误差
def compute_error(b, k, train_ids):
    total_error = 0
    for x_data1, y_data1 in train_ids:
        total_error += (y_data1 - (k * x_data1 + b)) ** 2
    return total_error / float(400) / 2.0


# 梯度下降法寻找最小值
def gradient_descent_runner(train_ids, b, k, lr, epochs):
    # 计算数据点总数
    m = 100
    # 循环迭代次数
    for i in range(epochs):
        b_gradient = 0
        k_gradient = 0
        # 计算梯度
        for x_data1, y_data1 in train_ids:
            b_gradient += -(1 / m) * (y_data1 - (k * x_data1 + b))
            k_gradient += -(1 / m) * (y_data1 - (k * x_data1 + b)) * x_data1
        # 使用梯度和学习率更新参数
        b = b - (lr * b_gradient)
        k = k - (lr * k_gradient)

    return b, k


print("初始值：b = {0}，k = {1}，误差 = {2}".format(b, k, compute_error(b, k, train_ids)))
print("运行中...")
b, k = gradient_descent_runner(train_ids, b, k, lr, epochs)
print("经过{0}次迭代后：b = {1}，k = {2}，误差 = {3}".format(epochs, b, k, compute_error(b, k, train_ids)))

# 绘图

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r')
plt.show()