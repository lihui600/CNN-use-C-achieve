# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import mnist
import json

(X_train,y_train),(X_test,y_test) = mnist.load_data()
img_rows,img_cols = X_train[0].shape[0],X_train[0].shape[1]
X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)

X_train=X_train.astype('float32')/255.
X_test=X_test.astype('float32')/255.
n_classes = len(set(y_train))
y_train = to_categorical(y_train,n_classes)
y_test = to_categorical(y_test,n_classes)


model = Sequential()
model.add(Conv2D(filters=6,kernel_size=(5,5),padding="valid",activation="relu",input_shape=[28, 28, 1]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='relu'))

model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])
callbacks=[EarlyStopping(monitor='val_accuracy',patience=5)]
model.fit(X_train,y_train,batch_size=64,epochs=10,verbose=1,validation_split=0.2,callbacks=callbacks)
model.save("model1.h5")

loss,accuracy = model.evaluate(X_test,y_test)
print(loss,accuracy)


preds = model.predict(X_test)
#Save the model
# serialize model to JSON

print("Saved model to disk")


