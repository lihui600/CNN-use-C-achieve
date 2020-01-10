import h5py
import numpy as np
import struct
import tensorflow as tf

# 模型地址
MODEL_PATH = '/home/lih/手写体数字识别model/model4.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    dense_4 = f['/model_weights/dense_4/dense_4']
    bias =  dense_4['bias:0'][:]
    weights = dense_4['kernel:0'][:]
   
    weight=tf.transpose(weights, perm=[1,0])
    print( weight)
    f = open('fc160_1.bin','wb')
    for i in range(120):
        for j in range(256):             
            f.write(struct.pack('f',weight[i][j]))
    for i in range(120):
        f.write(struct.pack('f',bias[i]))
    f.close()
print("读取完毕")