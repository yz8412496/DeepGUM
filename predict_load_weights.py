'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
import scipy.io as sio
import os
# added
import tensorflow as tf
from keras import backend as K

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.vgg16 import VGG16
from VGG16_rn import extract_XY_generator
from data_generator import load_data_generator, load_data_generator_simple
from data_generator import load_data_generator_List, load_data_generator_Uniform_List, rnEqui, rnHard, rnTra
from scipy.special import logsumexp
from log_gauss_densities import gausspdf, loggausspdf
from test import run_eval


# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.80
# sess = tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
start_time = time.time()
encode = "ISO-8859-1"
Age = "All"  # 'Y', 'M', 'O' and 'All'
DISPLAY_TEST = True
# DISPLAY_TEST=False
INDEP_MODE = False
PAIR_MODE = False
SAVE = False
UNI = False
DIAG = True
ISO = False
U_MIN_MAX = True
VALMODE = rnEqui

MNiso = True
MNdiag = False
MNinv = False

# ROOTPATH = 'C:/Users/Guest_admin/Downloads/GUMData/'
ROOTPATH = 'D:/CACD2000.tar/CACD2000/CACD2000/'
train_txt = 'trainingAnnotations.txt'
train_txt1 = 'trainingYoung.txt'
train_txt2 = 'trainingMiddle.txt'
train_txt3 = 'trainingOld.txt'
val_txt = 'validationAnnotations.txt'
val_txt1 = 'validationYoung.txt'
val_txt2 = 'validationMiddle.txt'
val_txt3 = 'validationOld.txt'
# test_txt = 'testAnnotations.txt'
# test_txt1 = 'testingYoung.txt'
# test_txt2 = 'testingMiddle.txt'
# test_txt3 = 'testingOld.txt'
test_txt = 'testAnnotations.txt'
test_txt1 = 'testAnnotations.txt'
test_txt2 = 'testAnnotations.txt'
test_txt3 = 'testAnnotations.txt'

LOW_DIM = 1
idOar = Age
argvs = ['-i', '-u']
# ROOTPATH=sys.argv[1]
# train_txt = sys.argv[2]
# test_txt = sys.argv[3]
# LOW_DIM = int(sys.argv[4])
ssRatio = 1.0  # float(sys.argv[3])/100.0
# idOar=sys.argv[5]
# to modify according to the task. A different evaluation function (test.py) will be used depending on the problem
PB_FLAG = "PROBLEM"

print(PB_FLAG)

for idarg, arg in enumerate(argvs):
    if arg == '-u':
        UNI = True
    elif arg == '-i':
        INDEP_MODE = True
    elif arg == '-p':
        PAIR_MODE = True
        DIAG = False
    elif arg == '-d':
        DIAG = True
    elif arg == '-iso':
        ISO = True
        DIAG = False
    elif arg == '-rnTra' or arg == '-rntra':
        VALMODE = rnTra
    elif arg == '-rnHard' or arg == '-rnhard':
        VALMODE = rnHard
    elif arg == '-reEqui'or arg == '-reequi':
        VALMODE = rnEqui
    elif arg == '-MNiso':
        MNiso = True
    elif arg == '-MNdiag':
        MNdiag = True
        MNiso = False
    elif arg == '-MNinv':
        MNinv = True
        MNiso = False

FEATURES_SIZE = 512
HIGH_DIM = FEATURES_SIZE

MAX_ITER_EM = 100
ITER = 2
# ITER = 1
WIDTH = 224
BATCH_SIZE = 64
# BATCH_SIZE = 1
NB_EPOCH = 15
PATIENCE = 1
NB_EPOCH_MAX = 50
LEARNING_RATE = 1e-04
# validationRatio = 0.80
validationRatio = 1.0
fileWInit = ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"


class MixtureModel:
    ''' Class of forward model'''

    def __init__(self):
        model_vgg16 = VGG16(input_shape=(224, 224, 3),
                            include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        x = Flatten(name='flatten')(model_vgg16.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bm1')(x)
        x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
        model = Model(model_vgg16.input, x)

        self.network = model

        # start_time_training = time.time()
        self.fileWbest = ROOTPATH+"Forward_Uni_best"+PB_FLAG+"_"+idOar+"_weights.hdf5"
        print("Training Forward")

        '''Fine tune the network according to our custom loss function'''

        # layer_nb=16 # number of finetunned layer
        # # train only some layers
        # for layer in self.network.layers[:layer_nb]:
        #     layer.trainable = False
        # for layer in self.network.layers[layer_nb:]:
        #     layer.trainable = True
        # self.network.layers[-1].trainable = True

        # compile the model
        # sgd = SGD(lr=learning_rate,
        #           momentum=0.9,
        #           decay=1e-06,
        #           nesterov=True)

        self.network.summary()

        self.network.load_weights(self.fileWbest)
        # self.network.compile(optimizer=sgd,
        #                      loss='mse')

    def predict(self, generator, n_predict):
        '''Generates output predictions for the input samples,
           processing the samples in a batched way.
        # Arguments
            generator: input a generator object.
            batch_size: integer.
        # Returns
            A Numpy array of predictions and GT.
        '''
        '''Extract VGG features and data targets from a generator'''

        i = 0
        Ypred = []
        Y = []
        for x, y in generator:
            if i >= n_predict:
                break
            pred = self.network.predict_on_batch(x)
            Ypred.extend(pred)
            Y.extend(y)
            i += len(y)

        return np.asarray(Ypred), np.asarray(Y)

    def evaluate(self, generator, n_eval, l=WIDTH, pbFlag=PB_FLAG):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            generator: input a generator object.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''

        Ypred, Y = self.predict(generator, n_eval)
        with open(Age+"_objs.pkl", 'wb') as f:
            pickle.dump([Ypred, Y], f)
        f.close()
        dict = {}
        dict['Ypred'] = Ypred
        dict['Y'] = Y
        sio.savemat(Age+"_objs.mat", dict)
        # print ("Y: " + str(Y))
        # print ("Ypred: " + str(Ypred))
        run_eval(Ypred, Y, l, pbFlag)


def readT(rootpath, file_train):
    return open(rootpath+file_train, 'r', encoding=encode).readlines()


if __name__ == '__main__':

    if Age == "Y":
        forward_Model1 = MixtureModel()
        (_, _), (gen_test, N_test) = load_data_generator(ROOTPATH,
                                                         train_txt1[:], test_txt1, validation=1.0, subsampling=ssRatio)
        forward_Model1.evaluate(gen_test, N_test, WIDTH)
    elif Age == "M":
        forward_Model2 = MixtureModel()
        (_, _), (gen_test, N_test) = load_data_generator(ROOTPATH,
                                                         train_txt2[:], test_txt2, validation=1.0, subsampling=ssRatio)
        forward_Model2.evaluate(gen_test, N_test, WIDTH)
    elif Age == "O":
        forward_Model3 = MixtureModel()
        (_, _), (gen_test, N_test) = load_data_generator(ROOTPATH,
                                                         train_txt3[:], test_txt3, validation=1.0, subsampling=ssRatio)
        forward_Model3.evaluate(gen_test, N_test, WIDTH)
    else:
        forward_Model = MixtureModel()
        (_, _), (gen_test, N_test) = load_data_generator(ROOTPATH,
                                                         train_txt[:], test_txt, validation=1.0, subsampling=ssRatio)
        forward_Model.evaluate(gen_test, N_test, WIDTH)
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)
