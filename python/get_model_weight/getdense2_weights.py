import h5py
import numpy as np
import struct
import tensorflow as tf

# 模型地址
MODEL_PATH = '/home/lih/手写体数字识别model/model4.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    dense_5 = f['/model_weights/dense_5/dense_5']
    bias =  dense_5['bias:0'][:]
    weights = dense_5['kernel:0'][:]
    weight=tf.transpose(weights, perm=[1,0])
    print( weight)
    print(weight[6][1])
    f = open('fc160_2.bin','wb')
    for i in range(84):
        for j in range(120):             
            f.write(struct.pack('f',weight[i][j]))
    for i in range(84):
        f.write(struct.pack('f',bias[i]))
    f.close()
print("读取完毕")