import random
from icecream import  ic
import numpy as np
import node,ope,layer,loss,optimizer,metrics
from graph import  default_graph

"""
测试目的：
    展示本框架如何搭建循环神经网络
测试过程：
    对应论文4.7.1
    人工构造两种不同类型的序列,搭建RNN来分类它们.
"""

def get_data(max_seq_length=10,min_seq_length=3,n_samples=1000,max_v=10):
    # x1是有序数列，x2是无序数列
    seq=[]
    label1 = np.array([1, 0]).reshape(1, -1)
    label2=np.array([0,1]).reshape(1,-1)
    x1 = np.array([float(i) / max_v for i in range(max_seq_length)])
    x2 = np.array([float(random.randint(0, max_v)) / max_v for i in range(max_seq_length)])

    for i in range((n_samples+1)//2):
        len=random.randint(min_seq_length,max_seq_length)
        start=random.randint(0,max_seq_length-len)

        #生成线性序列
        s1=x1[start:start+len]
        seq.append(np.c_[s1.reshape(1,-1),label1])

        s2=x2[start:start+len]
        seq.append(np.c_[s2.reshape(1, -1),label2])

    random.shuffle(seq)
    return seq

n_sampel=1000    #序列数目
seq_len=20      #序列的最大长度
#获取样本数据
data=get_data(n_samples=n_sampel,max_seq_length=seq_len,max_v=seq_len)

#确定结点
inputs=[node.Variable(shape=(1,1),init=False,trainable=False) for i in range(seq_len)]
U=node.Variable(shape=(seq_len,1),init=True,trainable=True)
W=node.Variable(shape=(seq_len,seq_len),init=True,trainable=True)
b=node.Variable(shape=(seq_len,1),init=True,trainable=True)
label=node.Variable((2,1),trainable=False)

hiddens=[]
h_old=None
cnt=0
for x in inputs:
    cnt=cnt+1
    h=ope.Add(ope.MatMul(U,x),b)
    if h_old is not None:
        h=ope.Add(ope.MatMul(W,h_old,name='node'+str(cnt)),h)
    h=ope.LeakyReLU(h)
    h_old=h
    hiddens.append(h)

#搭建模型
combing_point=ope.Combining()
y1=layer.fc(combing_point,seq_len,2,"None")
pred=ope.Logistic(y1)
loss=loss.CrossEntropyWithSoftMax(y1,label)
accuracy=metrics.Accuracy(pred,label)

lr=0.001
batch_size = 10
optimizer=optimizer.GradientDescent(default_graph,target=loss,learning_rate=lr)
for epoch in range(1000):
    for i in range(n_sampel):
        #输入不定长序列
        combing_point.combine( hiddens[data[i].shape[1]-2-1])
        for j in range(data[i].shape[1]-2):
            inputs[j].set_value(np.mat(data[i][:,j]))
        label.set_value(np.mat( data[i][:,-2:].reshape(label.shape())))
        #前向和反向传播
        optimizer.forward_backward()
        #更新参数
        if (i+1)%batch_size==0:
            optimizer.update()

    accuracy.init()
    for i in range(n_sampel):
        combing_point.combine( hiddens[data[i].shape[1]-2-1])
        for j in range(data[i].shape[1]-2):
            inputs[j].set_value(np.mat(data[i][:,j]))

        label.set_value(np.mat( data[i][:,-2:].reshape(label.shape())))
        accuracy.forward()
        loss.forward()

    loss.forward()
    print('epoch:',epoch)
    accuracy.print_value()



