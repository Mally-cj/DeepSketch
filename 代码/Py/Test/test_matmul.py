import numpy as np
import time
import sys
sys.path.append("..")
import from_other.cspeed as cspeed

"""
测试目的：
    测试用3种方式计算矩阵乘法的效率
测试过程：
    3种方式分别是用python，用C++写的模块，用CUDA写的并行模块。
    打印时间，最后会验证结果是否一样
"""

#生成随机矩阵
aa=300
bb=200
cc=aa
a=np.random.randint(2,4,(aa,bb))
b=np.random.randint(2,4,(bb,cc))

#测试1
T1 = time.perf_counter()
gpu=0
c1=cspeed.matmul(a,b,gpu)
T2 =time.perf_counter()
print('算法1（朴素C++）——程序运行时间:%s毫秒' % ((T2 - T1)*1000))

#测试2
T1 = time.perf_counter()
gpu=1
c2=cspeed.matmul(a,b,gpu)
T2 =time.perf_counter()
print('算法2（gpu）——程序运行时间:%s毫秒' % ((T2 - T1)*1000))

#测试3
T1 = time.perf_counter()
c3=np.matmul(a,b)
T2 =time.perf_counter()
print('算法3(numpy)——程序运行时间:%s毫秒' % ((T2 - T1)*1000))

#确保计算结果相同
print((c1==c2).all())
print((c1==c3).all())
# print("c1",c1)
# print("c3",c3)