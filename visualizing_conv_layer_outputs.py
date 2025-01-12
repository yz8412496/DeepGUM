import time
import sys
import numpy as np
import pickle as pickle
import scipy.io
import os
# added
import tensorflow as tf
from keras import backend as K

from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

LOW_DIM = 1
LEARNING_RATE = 1e-04
ROOTPATH = 'D:/CACD2000.tar/CACD2000/CACD2000/'
PB_FLAG = "PROBLEM"
Age = "O" # 'Y', 'M', 'O' and 'All'
idOar = Age
L = 56

model_resnet50 = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
for layer in model_resnet50.layers[:54]:
    layer.trainable = False
x = Flatten(name='flatten')(model_resnet50.output)
x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
model = Model(model_resnet50.input, x)
# rn = Input(shape=(LOW_DIM,))
# weightedRn = merge([rn,x],mode='mul')
# modelRn = Model(input=[model_resnet50.input,rn], output=weightedRn)

# sgd = SGD(lr=LEARNING_RATE, momentum=0.9, decay=1e-06, nesterov=True)
# model.compile(optimizer=sgd, loss='mse')
# modelRn.compile(optimizer=sgd, loss='mse')

# model.summary()
# modelRn.summary()

fileWbest = ROOTPATH+"Forward_Uni_best"+PB_FLAG+"_"+idOar+"_weights.hdf5"
model.load_weights(fileWbest)

# count = 0
# for i in range(len(model.layers)):
#     layer = model.layers[i]
#     # check for convolutional layer
#     # if 'res' in layer.name and 'branch' in layer.name and i>=54:
#     if 'act' in layer.name and i>=54:
#         print(i, layer.name, layer.output.shape)
#         count = count + 1

layer = model.layers
filters, biases = layer[L].get_weights()
print(layer[L].name, filters.shape)

# normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)

# fig1 = plt.figure(figsize=(8, 12))
# columns = 8
# rows = 8
# n_filters = columns * rows
# for i in range(1, n_filters + 1):
#     f = filters[:, :, :, i-1]
#     fig1 = plt.subplot(rows, columns, i)
#     fig1.set_xticks([])
#     fig1.set_yticks([])
#     plt.imshow(f[:, :, 0], cmap='gray')
# plt.show()

# conv_layer_index = [59, 111, 173]
conv_layer_index = [56, 170, 173, 174, 175, 176]
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(input=model.input, output=outputs)
# model_short.summary()

# img_name = '18_Rihanna_0005.jpg'
# img_name = '35_Jack_Davenport_0010.jpg'
# img_name = '60_Michael_Bowen_0001.jpg'
# img = load_img(ROOTPATH+img_name, target_size=(224, 224))

PATH = 'C:/Users/Guest_admin/Downloads/'
img_name = 'S_20%_19_Remove_Outliers.png'
img = load_img(PATH+img_name, target_size=(224, 224))
img_arr = img_to_array(img)

img_arr = np.expand_dims(img_arr, axis=0)
feature_output = model_short.predict(img_arr)
# feature_output = model.predict(img_arr)

columns = 8
rows = 8
n_filters = columns * rows
for ftr in feature_output:
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, n_filters + 1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        # plt.imshow(ftr[:, :, i-1], cmap='gray')
plt.show()

model.summary()
