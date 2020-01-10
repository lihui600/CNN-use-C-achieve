# -*- coding: utf-8 -*-

from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import cv2
from matplotlib import pyplot as plt

img_path = "/home/lihui/手写体数字识别model/picture/4-3.jpg"
img = image.load_img(img_path,target_size=(28,28))
img = img.convert('L')

x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
x=x.astype('float32')/255.

model = load_model('/home/lihui/手写体数字识别model/手写体识别model3/model3.h5') 
preds = model.predict(x)
print(preds)
n_examples=1
plt.figure(figsize=(10,10))
for i in range (n_examples):
    ax=plt.subplot(2,n_examples,i+1)
    plt.imshow(x[i, :, :, 0],cmap='gray')
    plt.title("Predicted:{}".format(np.argmax(preds[i])))
    plt.axis('off')
plt.show()
