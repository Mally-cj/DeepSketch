'''
测试目的：
    测试Tranier可以用来快速训练
测试过程：
    搭建LeNet模型，用Trainer训练
'''
import sys
# sys.path.append('../../')
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
import node,ope,optimizer,layer,loss,trainer,metrics
import numpy as np
from graph import  default_graph
from  icecream import  ic
def make_one_hot(label):
    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(label.reshape(-1, 1))
    return one_hot_label

def normalize(X):
    return X/ 255


filepath="../Example/data/MNIST/raw/"
# ic(filepath)

X=np.load(filepath+"X_train.npy")[:1000]
y=(np.load(filepath+"label_train.npy").T)[:1000]

X=normalize(X)
one_hot_label=make_one_hot(y)


img_shape = (28, 28)
x = node.Variable(img_shape, init=False, trainable=False)
one_hot = node.Variable(shape=(10, 1), init=False, trainable=False)

# 卷积层
conv1 = layer.conv([x], img_shape, 6, (5, 5), "ReLU")
#池化层
pooling1 = layer.pooling(conv1,kernel_shape=(2, 2), stride=(2, 2))


conv2 = layer.conv(pooling1, (14, 14), 16, (5, 5), "LeakyReLU")
pooling2 = layer.pooling(conv2,kernel_shape= (2, 2), stride=(2, 2) )
fc1 = layer.fc( ope.Concat(*pooling2), 784, 120, "LeakyReLU")
fc2 = layer.fc(fc1, 120, 84, "LeakyReLU")
output = layer.fc(fc2, 84, 10,"None")
predict =ope.SoftMax(output)
loss =loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.01

# 优化器
optimizer = optimizer.Adam(default_graph, loss, learning_rate)

accuracy =metrics.Accuracy(predict, one_hot)

# 批大小
batch_size = 32

trainer = trainer.Trainer(input_node=x, label_node=one_hot, loss_node=loss,
    optimizer=optimizer, metrics=accuracy,
    eval_on_train_bool=True,epoches=30,
    batch_size=batch_size,sample_number=10)

trainer.train_and_eval(input_data=X, label_data=one_hot_label)


ic('yes')
