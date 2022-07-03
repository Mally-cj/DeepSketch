'''
测试目的：
    测试Tranier可以用来快速训练
测试过程：
    搭建LeNet模型，用Trainer训练

测试目的：
    卷积神经网络训练索贝尔滤波器，验证该框架可以实现卷积神经网络
测试过程：
    对应论文4.7.2
    使用Trainer来训练
    filter 是要训练的滤波器， 它的初始值是 0 到 1 之间的随机数，img 是一张图片，sobel_img 是被索贝尔滤波器滤过后并且都乘以-1 的二维矩阵，
    故而 Add 其实表示的是 img 经 filter 滤波 后的二维图像与用真正的索贝尔滤波器训练的二维图像的差值。
'''
import numpy as np
import  matplotlib.pyplot as plt
import node,ope,optimizer,matplotlib
import from_other.cspeed as cspeed
from icecream import  ic
from graph import  default_graph
import trainer
#对图片与真正的sobel滤波器做卷积，得到卷积结果B
pic= np.mat(matplotlib.image.imread('lena.jpg')/255)
pic_size=pic.shape[0]*pic.shape[1]
sobel=np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
ic("图片完成")
aa= cspeed.C_Convolve()
sobel_img= np.mat(aa.compute(pic,sobel,0))
print('第一次卷积')

# plt.imshow(sobel_img)
sobel_img= sobel_img*(-1)

# ic(type(pic),pic.shape)

#得到y
img=node.Variable(shape=(pic.shape[0],pic.shape[1]),init=False,trainable=False)
img.set_value(np.mat(pic))
filter=node.Variable(shape=(3,3),init=True,trainable=True)
B=node.Variable(shape=(pic.shape[0],pic.shape[1]),init=False,trainable=False)
B.set_value(sobel_img)
y=ope.Add(ope.Convolve(img,filter),B)

#
N=node.Variable((1,1),init=False,trainable=False)
N.set_value(np.mat(1.0/pic_size))
loss=ope.MatMul( ope.MatMul( ope.Reshape(y,shape=(1,pic_size)),ope.Reshape(y,shape=(pic_size,1)) ),N )

learing_rate=0.01
# optimizer=optimizer.GradientDescent(default_graph,loss,learning_rate=learing_rate)
optimizer=optimizer.Adam(default_graph,loss,learning_rate=learing_rate)

batch_size=1
trainer = trainer.Trainer(
    input_node=img, label_node=None, loss_node=loss, optimizer=optimizer,
    eval_on_train_bool=False,epoches=100, batch_size=batch_size,sample_number=1)

# 把pic转换成训练器能接收的格式，即第一维表示的是样本的序列
pic=np.array(pic)
new_pic=np.stack([pic]*2,axis=0)
ic(new_pic.shape)
trainer.train_and_eval(input_data=new_pic)

#看看训练出的滤波器效果
filter.forward()
ic(filter.value)
# plt.imshow(pic)
# plt.imshow(sobel_img,cmap='gray')
# plt.show()