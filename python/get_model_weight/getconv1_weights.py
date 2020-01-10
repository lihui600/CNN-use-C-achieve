# -*- coding: utf-8 -*-

import h5py
import numpy as np
import struct
import tensorflow as tf

# 模型地址
MODEL_PATH = '/home/lih/手写体数字识别model/model4.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    conv2d_3 = f['/model_weights/conv2d_3/conv2d_3']
    bias =  conv2d_3['bias:0'][:]
    weights = conv2d_3['kernel:0'][:]
    weight=tf.transpose(weights, perm=[3, 2,0, 1])
    print(weight)
    f = open('conv1.bin','wb')
    for i in range(6):
        for j in range(1):
                 for m in range(5):
                         for n in range(5):
                                 #print(weight[i][j][m][n])
                                f.write(struct.pack('f',weight[i][j][m][n]))
                                #print(weight[i][j][m][n])
    
    for i in range(6):
        f.write(struct.pack('f',bias[i]))
    f.close()
print("读取完毕")
