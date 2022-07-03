from  node import  Node
from abc import ABC, abstractmethod
import numpy as np
from icecream import ic
class Metrics(Node):
    """
    评估结点,对应论文3.3.5
    """
    def __init__(self,*parents,**kwargs):
        # 一般传入两个结点，即父亲，一个是预测值结点，一个是标签结点
        Node.__init__(self,*parents,**kwargs)
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abstractmethod
    def init(self):
        #每个具体子类的初始方法不同
        pass

    def get_jacobi(self,parent):
        #对于评估指标结点，不需要参加反向传播，所以计算雅可比矩阵没有意义
        raise  NotImplementedError()

    @staticmethod
    def prob_to_labelform(prob, thresholds=0.5):
        # 把预测值转化为预测数字标签形式
        if prob.shape[0]>1:
            labels = np.zeros((prob.shape[0], 1))
            #如果是多分类，预测类别为概率最大的类别
            #argmax选取列方向最大值的索引
            labels=np.argmax(prob,axis=0)
        else:
            # # 要做特殊处理，因为每个预测值和标签的比对关系是不一样的, 对于当前泰坦尼克号这个例子是这样的
            labels=np.where(prob<thresholds,-1,1)
        return labels

    @staticmethod
    def onehot_to_label(onehot, thresholds=0.5):
        # 把onehot形式的标签转化为数字标签
        if onehot.shape[0] > 1:
            labels = np.zeros((onehot.shape[0], 1))
            labels = np.argmax(onehot, axis=0)
        return labels


    # def value_str(self):
    #     return "{}: {:.4f} ".format(self.__class__.__name__, self.value)



class Accuracy(Metrics):
    def __str__(self):
        str='计算准确率的结点,accuracy'
        return str

    def init(self):
        self.correct_num=0
        self.total_num=0

    def compute(self):
        # 假设第一个父结点是预测值，第二个父结点是标签
        pred=Metrics.prob_to_labelform(self.parents[0].value)
        label=Metrics.onehot_to_label(self.parents[1].value)
        assert pred.shape==label.shape

        self.correct_num+= np.sum(pred==label)
        self.total_num+=len(pred)
        self.value=0

        if self.total_num!=0:
            self.value= float(self.correct_num)/self.total_num

    def print_value(self):
        self.forward()
        print('accuracy:',self.value)

class Precision(Metrics):
    # 查准率

    def init(self):
        self.pre_true_1_num=0  #预测为1并且正确的样本数量
        self.pre_1_num=0      #预测为1的样本数量

    def __str__(self):
        str = '计算查准率的结点,Precision'
        return str

    def compute(self):
        pred = Metrics.prob_to_labelform(self.parents[0].value)
        label = Metrics.onehot_to_label(self.parents[1].value)

        self.value = 0

        self.pre_true_1_num+=np.sum(pred==gt and pred==1)
        self.pre_1_num+=np.sum(pred==1)


        if self.pre_1_num != 0:
            self.value = float(self.pre_true_1_num) / self.pre_1_num

        def print_value(self):
            print('Precision:', self.value)
