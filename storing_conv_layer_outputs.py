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

start_time = time.time()
LOW_DIM = 1
LEARNING_RATE = 1e-04
ROOTPATH = 'D:/CACD2000.tar/CACD2000/CACD2000/'
FPATH = 'D:/Features/'
PB_FLAG = "PROBLEM"
test_txt_Y = 'testingYoung.txt'
test_txt_M = 'testingMiddle.txt'
test_txt_O = 'testingOld.txt'
encode = "ISO-8859-1"
test_list = [test_txt_Y, test_txt_M, test_txt_O]
Age_list = ['Y', 'M', 'O']

for iter_test in range(1, len(test_list)):
    for iter_age in range(0, len(Age_list)):
        test_txt = test_list[iter_test]
        Age = Age_list[iter_age]
        idOar = Age
        Pos = "Before_" # "Before_" or "After_"
        # L = 56

        if test_txt=='testingYoung.txt':
            Group = "GY_"
        elif test_txt=='testingMiddle.txt':
            Group = "GM_"
        else:
            Group = "GO_"

        if Age=="Y":
            Network = "NY"
        elif Age=="M":
            Network = "NM"
        else:
            Network = "NO"

        imFile = []
        imTest = open(ROOTPATH+test_txt, 'r', encoding=encode).readlines()
        for i,image in enumerate(imTest):
            currentline=image.strip().split(" ")
            imFile.append(currentline[0])

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

        # layer = model.layers
        # filters, biases = layer[L].get_weights()
        # print(layer[L].name, filters.shape)

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

        # img_name = '18_Rihanna_0005.jpg'
        # img_name2 = '35_Jack_Davenport_0010.jpg'
        # img_name3 = '60_Michael_Bowen_0001.jpg'
        # I = [img_name, img_name2, img_name3]

        # conv_layer_index = [56, 60, 63]
        if Pos == 'Before_':
            conv_layer_index = [56, 60, 63, 66, 70, 73, 76, 80, 83, 86, 87, 92, 95, 98, 102, 105, 108, 112, 115,
                                118, 122, 125, 128, 132, 135, 138, 142, 145, 148, 149, 154, 157, 160, 164, 167, 170]
        else:
            conv_layer_index = [55, 59, 62, 65, 69, 72, 75, 79, 82, 85, 91, 94, 97, 101, 104, 107, 111, 114, 117,
                                121, 124, 127, 131, 134, 137, 141, 144, 147, 153, 156, 159, 163, 166, 169, 173]

        outputs = [model.layers[i].output for i in conv_layer_index]
        model_short = Model(input=model.input, output=outputs)
        # model_short.summary()

        Q = []
        for i in range(0, len(imFile)):
            img = load_img(ROOTPATH+imFile[i], target_size=(224, 224))
            img_arr = img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            feature_output = model_short.predict(img_arr)
            R = []
            for ftr in feature_output:
                sum = []
                sum2 = []
                sum3 = []
                sum4 = []
                ftr_tmp = ftr[0, :, :, :]
                for c in range(ftr_tmp.shape[2]):
                    ftr_chnnl = ftr_tmp[:, :, c]
                    sum.append(np.average(np.square(ftr_chnnl)))
                    sum2.append(np.average(np.abs(ftr_chnnl)))
                    sum3.append(np.max(ftr_chnnl))
                    sum4.append(np.average(ftr_chnnl))
                R.append([sum, sum2, sum3, sum4])
            Q.append(R)

        # print("saving .pkl")
        # fname = Pos + Group + Network + ".pkl"
        # open_file = open(FPATH+fname, 'wb')
        # pickle.dump(Q, open_file)
        # open_file.close()

        # feature_output = model.predict(img_arr)

        # columns = 8
        # rows = 8
        # n_filters = columns * rows
        # for ftr in feature_output:
        #     fig = plt.figure(figsize=(12, 12))
        #     for i in range(1, n_filters + 1):
        #         fig = plt.subplot(rows, columns, i)
        #         fig.set_xticks([])
        #         fig.set_yticks([])
        #         plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #         # plt.imshow(ftr[:, :, i-1], cmap='gray')
        # plt.show()

        print("saving .mat file")
        fname = Pos + Group + Network
        dict={}
        dict['Q'] = Q
        scipy.io.savemat(FPATH+fname+".mat",dict)

end_time = time.time()
print("Time elapsed: ", end_time-start_time)
model.summary()






