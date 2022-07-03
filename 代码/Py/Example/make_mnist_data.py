# coding: utf-8

"""
内容：
    下载mnist数据,并处理好
"""

import os.path
import gzip
import pickle
import os
import numpy as np
from icecream import ic
url_base = 'http://yann.lecun.com/exdb/mnist/'  # mnist官网，下载失败可以从此处下载，文件名见下方字典
key_file = {  # 字典存储下载好的文件名
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__)返回此片代码绝对路径，os.path.dirname（）返回此路径文件的目录
save_file = dataset_dir + "\\mnist.pkl"
train_num = 60000  # mnist数据集共60000个数据用于训练，10000个数据用于测试
test_num = 10000
img_size = 784

def _load_img(file_name):
    file_path = dataset_dir + "\\data\\MNIST\\raw\\" + file_name
    # print("file_path:",file_path)
    print("Converting " + file_path + " to NumPy Array ...")

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)  # 共image_siza=28*28列，-1为自适应行数
    # print("Done",file_name)
    return data


def _load_label(file_name):
    file_path = dataset_dir + "\\data\\MNIST\\raw\\" + file_name

    print("Converting " + file_path + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    # print("Done")
    return labels


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    print('conver sucess')
    return dataset


def init_mnist():
    print('here')
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=False, one_hot_label=False):
    """读入MNIST数据集
    """
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)


    return (dataset['train_img'], dataset['train_label'].T), (dataset['test_img'], dataset['test_label'].T)


(x_train, label_train), (x_test,label_test) = load_mnist(flatten=False, normalize=False)
# ic(type(x_train),x_train.shape)
# ic(type(label_train),label_train.shape)
# ic(type(x_test),x_test.shape)
# ic(type(label_test),label_test.shape)

x_train=np.mat(x_train)   ## X.shape:(1000.784)
label_train=np.mat(label_train).astype(np.int32)
x_test=np.mat(label_test)
label_test=np.mat(label_test).astype(np.int32)


# y=np.mat(y)   ## y.shape: (1, 1000)


filepath="./data/MNIST/raw/"
np.save(filepath+"X_train.npy",x_train)
np.save(filepath+"label_train.npy",label_train)
np.save(filepath+"X_test",x_test)
np.save(filepath+"label_test",label_test)
