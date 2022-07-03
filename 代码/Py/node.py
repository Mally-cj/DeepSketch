# node.py
from  graph import default_graph
import numpy as np
from abc import ABC, abstractmethod
from icecream import ic
import from_other.cspeed as cspeed

useCplus=False
useGPU = 1


class Node(object):
    """
    基类结点,对应论文3.3.1
    """
    def __init__(self,*parents,**kwargs):
        #把结点加入默认的图
        self.graph=default_graph
        self.graph.add_node(self)

        # 生成新结点的名字
        self.kwargs = kwargs
        self.gen_node_name(self)

        #性质
        self.parents=list(parents)
        self.children=[]
        self.value=None
        self.jacobi=None

        for parent in self.parents:
            parent.children.append(self)

    @abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值.在前向传播时使用.
        """
        pass

    @abstractmethod
    def get_jacobi(self,parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵.在反向传播时用.
        """
        pass

    #以下为node的通用函数
    def dimension(self):
        assert self.value is not None
        return self.value.shape[0]*self.value.shape[1]

    def shape(self):
        assert self.value is not None
        return  self.value.shape

    def clear_jacobi(self):
        self.jacobi=None

    def reset_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()

    def forward(self):
        for parent in self.parents:
            if parent.value is None:
                parent.forward()

        self.compute()

    def backward(self,result):
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                # ic(result.dimension(), self.dimension())
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))

                for child in self.children:
                    # 这里有个乘法顺序要注意，因为都是对w求导且w在左边（y=wx+b），故上个结点对该结点的导数是在右边的
                    if child.value is not None:
                        if useCplus:
                            a=child.backward(result)
                            b=child.get_jacobi(self)
                            self.jacobi+=cspeed.matmul(a,b,useGPU)
                        else:
                            self.jacobi += child.backward(result) * child.get_jacobi(self)
        return self.jacobi

    def gen_node_name(self,kwargs):
        self.name = self.kwargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.nodes_count()))


class Variable(Node):
    """
    变量结点,对应论文3.3.2

    """
    def __init__(self,shape,trainable=False,init=False, **kwargs):
        Node.__init__(self, **kwargs)
        self.size=shape

        #变量结点可否训练
        self.trainable=trainable

        #如果要初始话，就用随机值赋值
        if init:
            self.set_value(np.mat(np.random.normal(0, 0.001, self.size)))

    def shape(self):
        return self.size

    def set_value(self,value):
        assert isinstance(value, np.matrix)
        # 注意Variable的形状是在定义的时候就设定好的，故而使用self.size
        assert self.size == value.shape

        self.reset_value()
        self.value=value






















