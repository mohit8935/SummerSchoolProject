#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:27:54 2018

@author: MohitNihalani
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Flatten,Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV



from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
x_train  = np.reshape(x_train,x_train[48,48,3])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255    
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print ('hello')
model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=32, verbose=0)
print('hello2')
filter_size = [(5,5)]
print('hello3')
param_grid = dict(filter_size=filter_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
print('hello4')
grid_result = grid.fit(x_train, y_train)
print('hello5')
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
