import from_other.cspeed as cspeed
import numpy as np
import time
from icecream import ic
"""
测试目的：
    测试用3种方式计算卷积的效率
测试过程：
    3种方式分别是用python，用C++写的模块，用CUDA写的并行模块。
    打印时间，最后会验证结果是否一样
"""

def conv(data,kernel):
    # python写的卷积
    w, h = data.shape
    kw, kh = kernel.shape
    hkw, hkh = int(kw / 2), int(kh / 2)
    pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)) )
    padded = np.mat(np.zeros((pw, ph)))
    padded[hkw:hkw + w, hkh:hkh + h] = data

    value = np.mat( (np.zeros((w, h))))
    for i in np.arange(0, w):
        for j in np.arange(0, h):
           value[i, j] = np.sum(np.multiply(padded[i:i + kw, j:j + kh], kernel))
    return value


#生成随机矩阵
data=np.random.randint(0,1,(28,28))
kernel=np.random.randint(0,1,(2,5))

#测试1
T1 = time.perf_counter()
c1=conv(data,kernel)
T2 =time.perf_counter()
print('算法1（使用python）——程序运行时间:%s毫秒' % ((T2 - T1)*1000))


#测试2
T1 = time.perf_counter()
cc2=cspeed.C_Convolve()
gpu=False
c2=cc2.compute(data,kernel,gpu)
T2 =time.perf_counter()
print('算法2（使用C++）——程序运行时间:%s毫秒' % ((T2 - T1)*1000))

#测试3
T1 = time.perf_counter()
cc3=cspeed.C_Convolve()
gpu=True
c3=cc3.compute(data,kernel,gpu)
T2 =time.perf_counter()
print('算法3（使用C++和cuda）——程序运行时间:%s毫秒' % ((T2 - T1)*1000))

#确保计算结果相同
ic((c1==c2).all())
ic((c1==c3).all())

