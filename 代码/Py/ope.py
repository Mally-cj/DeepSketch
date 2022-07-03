from node import Node
import numpy as np
import example
from icecream import ic
import from_other.cspeed as cspeed
from graph import default_graph
useCplus=True
useGPU = False
class Operator(Node):
    """
    运算结点,对应论文3.3.3
    """
    pass

class Add(Operator):
    def compute(self):
        assert self.parents is not None

        self.value=np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            assert self.shape() ==parent.shape()
            self.value+=parent.value

    def get_jacobi(self, parent):
        return np.identity(parent.dimension())

class MatMul(Operator):
    def __init__(self, *parents, **kwargs):
        Node.__init__(self, *parents, **kwargs)

    def compute(self):
        assert len(self.parents)==2
        assert self.parents[0].shape()[1] ==self.parents[1].shape()[0]

        # 因为这里如果用C加速总是会容易报错，所以先用useCplus1控制开关
        useCplus1=0

        if useCplus1:
            self.value=cspeed.matmul(self.parents[0].value , self.parents[1].value,useGPU)
        else:
            self.value=np.matmul(self.parents[0].value ,self.parents[1].value)

    def get_jacobi(self,parent):
        # 矩阵乘法的雅可比矩阵在书41页
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)

        elif parent is self.parents[1]:
            """
               原理比较复杂，最后是通过重排列行和列的索引方式实现的
            """
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class Step(Operator):
    def compute(self):
        assert len(self.parents)==1
        self.value=np.mat(np.where(self.parents[0].value>0.0,1.0,0.0))

    def get_jacobi(self,parent):
        """
        因为在这个代码里，Step只用于得到预测值，故而不用对其反向传播求导就不算了
        """
        pass

class Multiply(Operator):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
            self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)

class Logistic(Operator):
    def __init__(self, *parents, **kwargs):
        Node.__init__(self, *parents,**kwargs)
        self.cc=cspeed.C_Logistic()
    def compute(self):
        if useCplus:
            self.value=self.cc.compute( self.parents[0].value,useGPU)
        else:
            x = self.parents[0].value
            x=np.power(np.e, np.where(x <-100, 100, -x))
            self.value = np.mat(1.0 / (1.0 + x))

    def get_jacobi(self, parent):
        if useCplus:
            return  self.cc.getjacobi()
        else:
            return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)

class SoftMax(Operator):
    """
    SoftMax函数
    对应论文4.3.1
    """

    @staticmethod
    def softmax(a):
        # 设置为静态方法的原因是在交叉熵的时候可以调用
        useCplus1=0
        if useCplus1:
            gpu=1
            return  np.mat( cspeed.C_SoftMax_compute(a,gpu) )
        else:
            a[a > 1e2] = 1e2  # 防止指数过大
            ep = np.power(np.e, a)
            return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        # 我们不实现SoftMax节点的get_jacobi函数， 因为它只用在输出预测值时。训练时使用CrossEntropyWithSoftMax节点
        raise NotImplementedError("Don't use SoftMax's get_jacobi")

class LeakyReLU(Operator):
    nslope=0.1
    def compute(self):
        self.value=np.mat(
            np.where(self.parents[0].value>0.0, self.parents[0].value,self.nslope*self.parents[0].value)
        )
    def get_jacobi(self,parent):
        return np.diag(np.where(self.parents[0].value.A1>0.0, 1.0,self.nslope ))

class Reshape(Operator):
    """
    改变父节点的值（矩阵）的形状
    对应论文4.4.2
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2


    def compute(self):
        # 因为使用Reshape时，都是对向量做转置，即n*1变到1*n，故而用np.reshape就可以了
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

class Concat(Operator):
    """
    将多个父节点的值连接成向量
    """

    def compute(self):
        assert len(self.parents) > 0
        # for a in self.parents:
        #     # ic(type(a.value))
        # 将所有父节点矩阵展平并连接成一个向量
        self.value = np.concatenate(
            [np.mat(p.value.flatten()) for p in self.parents],
            axis=1
        ).T



    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
               0:dimension] = np.eye(dimension)

        return jacobi

class Combining(Operator):
    """
    对应论文4.5.2
    """
    def compute(self):
        assert len(self.parents) == 1
        assert self.parents[0] is not None
        self.value = self.parents[0].value

    def get_jacobi(self, parent):

        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def combine(self, node):
        """
        将本节点焊接到输入节点上wwwwwwwwwwwwwwww
        """

        # 首先与之前的父节点断开

        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # 与输入节点焊接
        self.parents.append(node)
        node.children.append(self)

class Convolve(Operator):
    """
    对应论文4.6.2
    """
    def __init__(self,*parents,**kwargs):
        assert len(parents)==2
        Operator.__init__(self,*parents,**kwargs)
        self.cc=cspeed.C_Convolve()
        self.padded=None


    def compute(self):
        data=self.parents[0].value
        kernel=self.parents[1].value
        if useCplus:
            gpu=0
            # data=data.reshape(data.shape[0],data.shape[1],order='C')
            # kernel=kernel.reshape(kernel.shape[0],kernel.shape[1],order='C')
            self.value=np.mat(self.cc.compute(data,kernel,gpu))
            # ic(type(self.value))
            self.padded=np.mat(self.cc.get_paddle())
            # print('c++',self.value)
        else:
            w,h=data.shape
            kw,kh=kernel.shape
            hkw,hkh=int(kw/2),int(kh/2)

            pw,ph= tuple(np.add(data.shape,np.multiply((hkw,hkh),2)))
            self.padded=np.mat(np.zeros((pw,ph)))
            # ic(self.padded.shape,pw)

            self.padded[hkw:hkw+w, hkh:hkh+h]=data
            self.value=np.mat((np.zeros((w,h))))
            for i in np.arange(0,w):
                for j in np.arange(0,h):
                    self.value[i,j]=np.sum(np.multiply(self.padded[i:i+kw,j:j+kh],kernel))

            # print('py kernal',kernel)
            # data = data.reshape(data.shape[0], data.shape[1], order='C')
            # kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], order='C')
            # ic(data)
            # print(self.value[0:5,0:5])
            # tem= self.cc.compute(data, kernel, 0)
            # ic(tem.shape)
            # ic(type(self.value))
            # self.padded = np.mat(self.cc.get_paddle())
            # print('c++', tem)
            # print((self.value==tem).all())
            # print((self.value[0:5,0:5]==tem[0:5,0:5]))

    def get_jacobi(self,parent):
        data=self.parents[0].value
        kernel=self.parents[1].value

        w, h = data.shape
        kw, kh = kernel.shape
        hkw, hkh = int(kw / 2), int(kh / 2)
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi=[]
        if parent is self.parents[0]:
            for i in np.arange(hkw,hkw+w):
                for j in np.arange(hkh,hkh+h):
                    mask=np.mat(np.zeros((pw,ph)))
                    mask[i-hkw:i-hkw+kw, j-hkh:j-hkh+kh]=kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)

class MaxPooling(Operator):
    """
    最大值池化
    对应论文4.6.4
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.stride = kargs.get('stride')
        assert self.stride is not None

        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(
                    np.max(window)
                )

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)

                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)
        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):

        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag

class ScalarMultiply(Operator):
    """
    用标量（1x1矩阵）数乘一个矩阵
    """


    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]

# 一些函数补充

# 一些函数补充


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0]/filler.shape[0] == to_be_filled.shape[1]/filler.shape[1]
    h,w=filler.shape
    n=to_be_filled.shape[0]//filler.shape[0]
    for i in range(n):
        to_be_filled[i*h:(i+1)*h,i*w:(i+1)*w]=filler

    return to_be_filled