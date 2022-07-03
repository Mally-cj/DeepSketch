from abc import ABC, abstractmethod
import numpy as np
from icecream import ic
import time
import os
class Trainer(object):
    """
    对应论文4.6.2
    """
    def __init__(self,input_node,label_node,loss_node,optimizer,epoches,metrics=None,eval_on_train_bool=False,
                 eval_interval_epochnum=1,
                 batch_size=8,sample_number=None,*args,**kargs):
        #本训练器只处理只有一个输入结点的情况，故input就是输入结点
        self.input=input_node
        self.label=label_node
        self.epoches=epoches
        self.epoch=0
        self.batch_size=batch_size
        self.loss=loss_node
        self.optimizer=optimizer
        self.metrics=metrics
        self.eval_on_train_bool=eval_on_train_bool
        self.sample_number=sample_number
        self.eval_interval_epochnum=eval_interval_epochnum     #评估的间隔时间


    def train_and_eval(self,input_data=None,label_data=None):
        #确保第一个维度是表示样本序号的
        assert input_data.shape[0]==input_data.shape[0]
        if self.sample_number is None: self.sample_number = len(input_data)

        for self.epoch in range(self.epoches):
            # ic(self.epoch)
            self.train(input_data=input_data,label_data=label_data)

            print('- Epoch [{}] train finished'.format(self.epoch + 1))
            print(time.strftime('%Y-%m-%d %H:%M:%S,run', time.localtime(time.time())))
            if (self.epoch+1)%self.eval_interval_epochnum==0:
                # self.loss.forward()
                print('loss:{}'.format(self.loss.value[0,0]))
                if self.eval_on_train_bool and label_data is not  None:
                    self.eval(input_data=input_data,label_data=label_data)


    def train(self,input_data,label_data):
        #一个迭代的循环


        #input_data的第一个变量是x，它的长度代表样本数，也作为循环数
        for i in range(self.sample_number):

            self.input.set_value( np.mat(input_data[i,:].reshape(self.input.shape())))
            if self.label is not None:
                self.label.set_value( np.mat(label_data[i,:]).reshape(self.label.shape()))
            self.optimizer.forward_backward()

            if (i+1)%self.batch_size ==0 :
                self.optimizer.update()



    def eval(self,input_data,label_data):
        # 对每个样本调用评价结点
        if self.metrics is not None: self.metrics.init()

        for i in range(self.sample_number):
            self.input.set_value(np.mat(input_data[i, :].reshape(self.input.shape())))
            self.label.set_value(np.mat(label_data[i, :]).T)
            self.metrics.forward()

        ic(self.epoch)
        self.metrics.print_value()






















