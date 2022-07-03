# 测试CNN网络效果
import numpy as np
import  matplotlib.pyplot as plt
import node,ope,optimizer,matplotlib
import from_other.cspeed as cspeed
from icecream import  ic
from graph import  default_graph
import time
"""
测试目的：
    卷积神经网络训练索贝尔滤波器，验证该框架可以实现卷积神经网络
测试过程：
    对应论文4.7.2
    filter 是要训练的滤波器， 它的初始值是 0 到 1 之间的随机数，img 是一张图片，sobel_img 是被索贝尔滤波器滤过后并且都乘以-1 的二维矩阵，
    故而 Add 其实表示的是 img 经 filter 滤波 后的二维图像与用真正的索贝尔滤波器训练的二维图像的差值。
"""

# 导入一张图片，用C++写的卷积模块做这个图片与索贝尔滤波器的卷积，卷积结果再乘上-1
pic= np.mat(matplotlib.image.imread('lena.jpg')/255)
pic_size=pic.shape[0]*pic.shape[1]
sobel=np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
# ic("卷积图片完成")
aa= cspeed.C_Convolve()
sobel_img= np.mat(aa.compute(pic,sobel,0))
sobel_img= sobel_img*(-1)

#得到y
img=node.Variable(shape=(pic.shape[0],pic.shape[1]),init=False,trainable=False)
img.set_value(np.mat(pic))
filter=node.Variable(shape=(3,3),init=True,trainable=True)
B=node.Variable(shape=(pic.shape[0],pic.shape[1]),init=False,trainable=False)
B.set_value(sobel_img)
out_img=ope.Convolve(img,filter)
y=ope.Add(out_img,B)

#
N=node.Variable((1,1),init=False,trainable=False)
N.set_value(np.mat(1.0/pic_size))
loss=ope.MatMul( ope.MatMul(ope.Reshape(y,shape=(1,pic_size)),ope.Reshape(y,shape=(pic_size,1)) ),N )

learing_rate=0.01
# optimizer=optimizer.GradientDescent(default_graph,loss,learning_rate=learing_rate)
optimizer=optimizer.Adam(default_graph,loss,learning_rate=learing_rate)

for epoch in range(1000):
    T1 = time.perf_counter()
    optimizer.forward_backward()
    optimizer.update()
    loss.forward()
    if loss.value[0,0]<0.004:break
    # ic(y.value)
    # ic(loss.value)
    T2 = time.perf_counter()
    print('epoch:{:d},loss:{:.6f},time:{}'.format(epoch,loss.value[0,0],(T2 - T1)*1000))

filter.forward()
ic(filter.value)
plt.imshow(sobel_img,cmap='gray')
# plt.imshow(out_img.value,cmap='gray')
plt.show()

"""
训练结果：
1.07399795 1.33202533 1.33197489 
0.27773194 0.00687009 −0.26171502 
−1.34263261 −1.34583759 − 1.07254422
"""