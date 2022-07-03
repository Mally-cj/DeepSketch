"""
demo：
    反向传播模型判别男女
    对应论文3.4.2
意义：
    验证反向传播模型搭建成功
"""

import numpy as np
from icecream import ic
import node,ope,loss,optimizer
from graph import  default_graph

def make_test():
    # 生产测试数据
    m_h = np.random.normal(171, 6, 500)
    f_h = np.random.normal(158, 5, 500)
    m_w = np.random.normal(70, 10, 500)
    f_w = np.random.normal(57, 8, 500)
    m_bfrs = np.random.normal(16, 2, 500)
    f_bfrs = np.random.normal(22, 2, 500)

    m_labels = [1] * 500
    f_labels = [-1] * 500

    train_set = np.array([np.concatenate((m_h, f_h)),
                          np.concatenate((m_w, f_w)),
                          np.concatenate((m_bfrs, f_bfrs)),
                          np.concatenate((m_labels, f_labels))
                          ]).T

    np.random.shuffle(train_set)
    return  train_set


if __name__=='__main__':
    train_set=make_test()
    x=node.Variable(shape=(3,1))
    w=node.Variable(shape=(1,3),trainable=True)
    b=node.Variable(shape=(1,1),trainable=True)
    label=node.Variable(shape=(1,1))

    w.set_value(np.mat(np.random.normal(0,0.001,(1,3))))
    b.set_value(np.mat(np.random.normal(0,0.001,(1,1))))


    y=ope.Add(ope.MatMul(w,x),b)
    predict=ope.Logistic(y)

    # loss=loss.PerceptionLoss(ope.Multiply(label,y))
    loss=loss.LogLoss(ope.Multiply(label,y))
    learning_rate=0.01

    optimizer=optimizer.Adam(default_graph,loss,learning_rate)
    # optimizer=optimizer.GradientDescent(default_graph, loss,learning_rate)

    cur_batch_size=0  #当前
    bacth_size = 10
    for epoch in range(1000):
        for i in  range(len(train_set)):
            # 输入数据
            x.set_value(np.mat(train_set[i, :-1]).T)
            label.set_value(np.mat(train_set[i,-1]))

            optimizer.forward_backward()
            cur_batch_size+=1

            if cur_batch_size==bacth_size:
                optimizer._update()
                cur_batch_size=0
            # break

        if epoch%10==0:
            pred = []
            for i in range(len(train_set)):
                # 输入数据
                x.set_value(np.mat(train_set[i, :-1]).T)
                predict.forward()
                pred.append(predict.value[0, 0])  # 模型的预测结果：1男，0女

            # 将1/0结果转化成1/-1结果，好与训练标签的约定一致
            pred=(np.array(pred)>0.5).astype(np.int32)*2-1

            accuracy = (train_set[:, -1] == pred).astype(np.int32).sum() / len(train_set)

            print("训练次数为:",epoch,"时，准确率为：",accuracy)

