#用Python跑LeNet模型的示范
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
import time
import os
"""
demo：
    搭建卷积神经网络 LeNet 模型识别手写数字
具体过程：
    对应论文4.7.3
    求准确率的时候是用metric结点
"""

def make_one_hot(label):
    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(label.reshape(-1, 1))
    return one_hot_label

def normalize(X):
    return X/ 255

filepath="./data/MNIST/raw/"

X=np.load(filepath+"X_train.npy")[:200]
y=(np.load(filepath+"label_train.npy").T)[:200]

X=normalize(X)
one_hot_label=make_one_hot(y)
img_shape = (28, 28)

# 输入图像
x = Variable(img_shape, init=False, trainable=False,name='in_x')
# One-Hot标签
one_hot = Variable(shape=(10, 1), init=False, trainable=False,name='one_hot')

# 第一卷积层
conv1 = layer.conv([x], img_shape, 6, (5, 5), "ReLU")
# 第一池化层
pooling1 = layer.pooling(conv1,kernel_shape=(2, 2), stride=(2, 2))
# # 第二卷积层
conv2 = layer.conv(pooling1, (14, 14), 16, (5, 5), "ReLU")
# # 第二池化层
pooling2 = layer.pooling(conv2,kernel_shape= (2, 2), stride=(2, 2) )

# 全连接层
fc1 = layer.fc( ope.Concat(*pooling2), 784, 120, "ReLU",name='fc1')
fc2 = layer.fc(fc1, 120, 84, "ReLU",name='fc2')

# 输出层
output = layer.fc(fc2, 84, 10,"None",name='ouput')

# 分类概率
predict =ope.SoftMax(output,name="predict")

# 交叉熵损失
loss =loss.CrossEntropyWithSoftMax(output, one_hot,name="loss")

# 学习率
learning_rate = 0.01

# 优化器
optimizer=optimizer.Adam(default_graph,loss,learning_rate)
# optimizer = optimizer.GradientDescent(default_graph, loss, learning_rate)

# 批大小
batch_size = 32
default_graph.draw()
# 训练
record=[]
for epoch in range(100):
    batch_count = 0
    T1 = time.perf_counter()
    for i in range(len(X)):

        feature = np.mat(X[i]).reshape(img_shape)
        label = np.mat(one_hot_label[i]).T
        x.set_value(feature)
        # ic(label.shape)
        one_hot.set_value(label)

        optimizer.forward_backward()
        batch_count += 1
        if batch_count >= batch_size:
            print(time.strftime('%Y-%m-%d %H:%M:%S,run', time.localtime(time.time())))
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0

    pred=[]
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    pred=pred.reshape(pred.shape[0],1)
    accuracy = (y== pred).astype(np.int32).sum() / len(X)

    T2 = time.perf_counter()
    # print('算法1（朴素C++）——程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
    print(time.strftime('%Y-%m-%d %H:%M:%S,run', time.localtime(time.time())))
    print("epoch: {:d}, accuracy: {:.3f},运行{:.6f}毫秒 ".format(epoch + 1, accuracy,(T2 - T1) * 1000))

    record.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    record.append("epoch: {:d}, accuracy: {:.3f},运行{:6f}毫秒 ".format(epoch + 1, accuracy,(T2 - T1) * 1000))


dataset_dir = os.path.dirname(os.path.abspath(__file__))

# 把每次跑的结果写在一个Record.txt里.
path=dataset_dir+"\\Record.txt"
file3 = open(path, 'w',encoding='UTF-8')
for i in record:
    file3.write(str(i)+'\n')
file3.close()
