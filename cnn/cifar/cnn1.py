#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:19:24 2018

@author: MohitNihalani
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard


batch_size = 32
epochs=150
num_classes = 10


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(256, (5, 5), padding='same',input_shape=(32,32,3),activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (1, 1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(256, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (2, 2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (1, 1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = num_classes,activation='softmax'))
"""
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units = 1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = num_classes,activation='softmax'))"""


# Compiling the CNN
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
from keras.preprocessing.image import ImageDataGenerator               
t_datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125)
t_datagen.fit(x_train)

tensorboard = TensorBoard(log_dir='/output/Graph', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit_generator(t_datagen.flow(x_train, y_train,
                                batch_size=batch_size),steps_per_epoch = x_train.shape[0]/batch_size,
                                epochs=100,
                                validation_data=(x_test, y_test),validation_steps = x_test.shape[0]/batch_size,
                                callbacks=[tensorboard]
                                )

scores = model.evaluate(x_test,y_test)
print(scores[0])
print(scores[1])
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

"""
xc=range(epochs)

/**plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('loss.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.savefig('acc.png')

plt.figure(3, figsize=(7,5))
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc')
plt.grid(True)
plt.savefig('val_acc.png')*//"""

#print plt.style.available # use bmh, classic,ggplot for big pictures

model.save_weights('/output/model.h5')
model.save('/output/model.hdf5')
"""
from keras.utils import plot_model
plot_model(model, to_file='/output/model.png')"""

