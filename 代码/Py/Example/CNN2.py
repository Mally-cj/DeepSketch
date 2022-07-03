"""
demo：
    搭建卷积神经网络 LeNet 模型识别手写数字
具体过程：
    对应论文4.7.3
    求准确率的时候没有用metric结点
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import node,ope,optimizer,loss
from node import Variable
from graph import  default_graph
from icecream import ic
import layer
from sklearn.datasets import make_circles
from scipy import signal
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

# 加载MNIST数据集，取一部分样本并归一化,然后转成矩阵，保存到本地
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# X, y = X[:1000] / 255, y.astype(np.int32)[:1000]
# X=np.mat(X)   ## X.shape:(1000.784)
# y=np.mat(y)   ## y.shape: (1, 1000)
# np.save("X.npy",X)
# np.save("y.npy",y)
X=np.load("X.npy")
y=np.load("y.npy")

ic(X.shape,y.shape)
# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))
ic(one_hot_label.shape)
# 输入图像尺寸
img_shape = (28, 28)

# 输入图像
x = Variable(img_shape, init=False, trainable=False,name='in_x')

# One-Hot标签
one_hot = Variable(shape=(10, 1), init=False, trainable=False,name='one_hot')

# 第一卷积层
conv1 = layer.conv([x], img_shape, 3, (5, 5), "ReLU")

# 第一池化层
pooling1 = layer.pooling(conv1, (3, 3), (2, 2))

# 第二卷积层
conv2 = layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

# 第二池化层
pooling2 = layer.pooling(conv2, (3, 3), (2, 2))

# 全连接层
fc1 = layer.fc(ope.Concat(*pooling2), 147, 120, "ReLU")

# 输出层
output = layer.fc(fc1, 120, 10, "None")

# 分类概率
predict =ope.SoftMax(output,name="predict")

# 交叉熵损失
loss =loss.CrossEntropyWithSoftMax(output, one_hot,name="loss")

# 学习率
learning_rate = 0.005

# 优化器
optimizer = optimizer.Adam(default_graph, loss, learning_rate)

# 批大小
batch_size = 32
default_graph.draw()
# 训练
for epoch in range(60):

    batch_count = 0

    for i in range(len(X)):

        feature = np.mat(X[i]).reshape(img_shape)
        label = np.mat(one_hot_label[i]).T
        x.set_value(feature)
        one_hot.set_value(label)

        optimizer.forward_backward()
        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    accuracy = (y == pred).astype(np.int32).sum() / len(X)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))