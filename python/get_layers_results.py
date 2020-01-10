# -*- coding: utf-8 -*-
from keras.preprocessing import image
import cv2
from keras import backend as K
from keras.models import load_model
#from PIL import Image
import numpy as np

def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]

model = load_model('/home/lih/手写体数字识别model/model4.h5')
img_path = "/home/lih/手写体数字识别model/picture/4-3.jpg"
img = image.load_img(img_path,target_size=(28,28))
img = img.convert('L')

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
input_x=x.astype('float32')/255.
conv1 = get_layer_output(model, input_x, index=0)
print(conv1.shape)
#print(conv1)
conv_1=conv1.reshape(6,1,24,24)
print(conv_1.shape)
#print(conv_1)

conv_1.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/conv1.txt', sep=',',format='%f')
pool1 = get_layer_output(model, input_x, index=1)
print(pool1.shape)
#print(pool1)
pool_1=pool1.reshape(6,1,12,12)
pool_1.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/pool1.txt', sep=',',format='%f')
print(pool_1.shape)
#print(pool_1)
conv2 = get_layer_output(model, input_x, index=2)
#print(conv2.shape)
conv_2=conv2.reshape(16,1,8,8)
#print(conv_2.shape)
conv_2.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/conv2.txt', sep=',',format='%f')
pool2 = get_layer_output(model, input_x, index=3)
print(pool2.shape)
#print(pool2)
pool_2=pool2.reshape(16,1,4,4)
pool_2.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/pool2.txt', sep=',',format='%f')
print(pool_2.shape)
#print(pool_2)
flatten1 = get_layer_output(model, input_x, index=4)
print(flatten1.shape)
flatten_1=flatten1.reshape(256)
flatten_1.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/flatten1.txt', sep=',',format='%f')
print(flatten_1.shape)
dense1 = get_layer_output(model, input_x, index=5)
print(dense1.shape)
dense_1=dense1.reshape(120)
dense_1.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/dense1.txt', sep=',',format='%f')
print(dense_1.shape)
#print(dense_1)
dense2 = get_layer_output(model, input_x, index=6)
print(dense2.shape)
dense_2=dense2.reshape(84)
dense_2.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/dense2.txt', sep=',',format='%f')
print(dense_1.shape)
output = get_layer_output(model, input_x, index=7)
print(output.shape)
output=output.reshape(10)
output.tofile('/home/lih/手写体数字识别model/手写体识别model4/param2_txt/output.txt', sep=',',format='%f')
print(output.shape)
#print(conv1)
