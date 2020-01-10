import h5py
import numpy as np
import struct
import tensorflow as tf

# 模型地址
MODEL_PATH = '/home/lih/手写体数字识别model/dense.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    dense_6 = f['/dense_6/dense_6']
    bias =  dense_6['bias:0'][:]
    weights = dense_6['kernel:0'][:]
    weights=weights
    weight=tf.transpose(weights, perm=[1,0])
    print(weights)
    print(bias)
    f = open('fc.bin','wb')
    for i in range(10):
        for j in range(84):             
            f.write(struct.pack('f',weight[i][j]))
    for i in range(10):
        f.write(struct.pack('f',bias[i]))
    f.close()
print("读取完毕")