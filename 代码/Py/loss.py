from node import Node
import numpy as np
from icecream import ic
import ope
import from_other.cspeed as cspeed
useCplus=False

class LossFunction(Node):
    """
    损失函数抽象类,对应论文3.3.4
    """
    pass
class PerceptionLoss(LossFunction):
    # 感知机损失函数
    def compute(self):
        """
              感知机损失，输入为正时为0，输入为负时为输入的相反数
        """
        x=self.parents[0].value
        self.value=np.mat(np.where(x>=0.0, 0.0, -x))

    def get_jacobi(self,parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())

class LogLoss(LossFunction):
    #对数损失函数
    def __init__(self, *parents, **kwargs):
        Node.__init__(self, *parents, **kwargs)
        self.cc = cspeed.C_LogLoss()   #使用C++模块,要先在C++内建立对应类

    def compute(self):
        assert len(self.parents) == 1
        if useCplus:
            gpu=1
            self.value=self.cc.compute(self.parents[0].value,gpu)
        else:
            x = self.parents[0].value
            self.value=np.log(1 + np.power(np.e, np.where(x <-100, 100, -x)))
            # 为了防止溢出，对指数进行了截断，其最大只能取到100

    def get_jacobi(self, parent):
        if useCplus:
            gpu=1
            return self.cc.getjacobi(parent.value,gpu)
        else:
            x = parent.value
            diag = -1 / (1 + np.power(np.e, np.where(x > 100, 100, x)))
            return np.diag(diag.ravel())

class CrossEntropyWithSoftMax(LossFunction):
    """
    交叉熵，对应论文4.3.2
    """
    def __init__(self, *parents, **kwargs):
        Node.__init__(self, *parents, **kwargs)
        self.cc = cspeed.C_CrossEntropyWithSoftMax()

    def compute(self):
        """
          对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
        """
        useCplus=0
        if useCplus:
            gpu=1
            self.value=self.cc.compute(self.parents[0].value,self.parents[1].value, gpu)
        else:
            prob = ope.SoftMax.softmax(self.parents[0].value)
            self.value = np.mat(
                -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))
    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = ope.SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T