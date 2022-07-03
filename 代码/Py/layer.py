from node import  Variable
import numpy as np
import ope
from icecream import ic
def fc(input,input_size,size,activation,name=None):
    """ 构建全连接层
    对应论文4.3.3
    :param input:
    :param input_size:输入的大小
    :param size: 输出的大小
    :param activation: 激活函数类型的名称
    :param name: 该结点的名称
   :return:
    """
    weights=Variable(shape= (size,input_size),init=True,name="fc_w")
    bias=Variable( shape=(size,1),init=True ,name='fc_b')
    affine=ope.Add(  ope.MatMul(weights,input,name="fc_MatMul"),bias)

    if activation =="LeakyReLU":
        return ope.LeakyReLU(affine)
    elif activation=="Logistic":
        return ope.Logistic(affine)
    else:
        return affine


def pooling(feature_maps, kernel_shape, stride):
    """构建池化层
    对应论文4.6.5
    :param feature_maps: 列表，包含多个输入特征图
    :param kernel_shape: 池化核的形状（宽和高）
    :param stride: 池化的步长,二维
    :return: 列表，包含多个池化后的特征图
    """
    outputs = []
    for fm in feature_maps:
        outputs.append(ope.MaxPooling(fm, size=kernel_shape, stride=stride))

    return outputs

def conv(feature_maps, input_shape, kernel_num, kernel_shape, activation):
    """构建卷积层
    对应论文4.6.3
    :param feature_maps: 列表，包含多个输入特征图
    :param input_shape: 特征图的形状（宽和高）
    :param kernels: 卷积核数量
    :param kernel_shape: 卷积核的形状（宽和高）
    :param activation: 激活函数类型
    :return: 列表,卷积后的结果
    """
    # 与输入同形状的全 1 矩阵
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for i in range(kernel_num):

        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = ope.Convolve(fm, kernel)
            channels.append(conv)

        channles = ope.Add(*channels,name='channels')
        bias = ope.ScalarMultiply(Variable((1, 1), init=True, trainable=True,name='bias_v'),ones,name="conv_matmul")
        affine = ope.Add(channles, bias,name='affine')

        if activation == "LeakyReLU":
            outputs.append(ope.LeakyReLU(affine))
        elif activation == "Logistic":
            outputs.append(ope.Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernel_num
    return outputs

