# -*- coding: utf-8 -*-

from keras.preprocessing import image
import cv2
from keras.models import load_model
import numpy as np
img_path = "/home/lih/手写体数字识别model/picture/4-3.jpg"
img = image.load_img(img_path,target_size=(28,28))
img = img.convert('L')

#x = image.img_to_array(img)
#print(x/255.0)
w,h= img.size
print(w,h)
f = open(img_path+'.bin','wb')


f.write(struct.pack('i',int(w)))
f.write(struct.pack('i',int(h)))

for i in range(w):
	for j in range(h):
		pixel_r = int(img.getpixel((i,j)))
		print(pixel_r)
		f.write(struct.pack('i',pixel_r))
