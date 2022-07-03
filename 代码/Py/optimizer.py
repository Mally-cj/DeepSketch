from node import Node,Variable
from graph import Graph
from abc import ABC, abstractmethod
import graph,node
from icecream import ic
import numpy as np
class Optimizer(object):
    """
    优化器
    对应论文 4.2.1
    """
    def __init__(self,graph,target):
        assert isinstance(target,Node)
        assert isinstance(graph,Graph)

        self.graph=graph
        self.target=target

        # 用于累加一个批大小的全部样本的梯度
        self.mp_gradient={}
        self.size_sample=0

    def get_gradient(self,node):
        #返回样本的平均梯度
        assert node in self.mp_gradient
        return self.mp_gradient[node]/self.size_sample


    @abstractmethod
    def _update(self):
        pass

    def update(self, var_gradients=None):
        # 执行更新
        self._update()

        # 清除累加梯度
        self.mp_gradient.clear()
        self.size_sample = 0


    def forward_backward(self):
        self.graph.clear_jacobi()
        self.target.forward()

        for node in self.graph.nodes:
            if isinstance(node,Variable) and node.trainable:
                node.backward(self.target)

                # # 最终结果（标量）对节点值的雅可比是一个行向量，其转置是梯度（列向量）
                # # 这里将梯度reshape成与节点值相同的形状，好对节点值进行更新。
                # gradient = node.jacobi.T.reshape(node.shape())

                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.mp_gradient:
                    self.mp_gradient[node]=gradient
                else :
                    # ic(self.mp_gradient[node].shape, gradient.shape)
                    self.mp_gradient[node]+=gradient

        self.size_sample+=1


class GradientDescent(Optimizer):
    """
    朴素梯度下降法
    对应论文4.2.2
    """
    def __init__(self,graph,target,learning_rate):
        Optimizer.__init__(self,graph, target)
        self.learning_rate=learning_rate


    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)
                # 用朴素梯度下降法更新变量节点的值
                node.set_value(node.value - self.learning_rate * gradient)


class Adam(Optimizer):
    """
    Adam优化器
    对应论文4.2.3
    """
    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1
        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2
        # 历史梯度累积
        self.v = {}
        # 历史梯度各分量平方累积
        self.s = {}

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient

                    # 各分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + 1e-10))

