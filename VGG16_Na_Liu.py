'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
import os
# added
import tensorflow as tf
from keras import backend as K

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.vgg16 import VGG16
from VGG16_rn import extract_XY_generator, extract_XY_generator_multiple
from data_generator_Na_Liu import load_data_generator,load_data_generator_simple
from data_generator_Na_Liu import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
from scipy.special import logsumexp
from log_gauss_densities import gausspdf,loggausspdf
from test import run_eval

start_time = time.time()
encode = "ISO-8859-1"
Age = "Na_Liu" # "C"
DISPLAY_TEST=True
# DISPLAY_TEST=False
INDEP_MODE=False
PAIR_MODE=False
SAVE=False
UNI=False
DIAG=True
ISO=False
U_MIN_MAX=True
VALMODE=rnEqui

MNiso=True
MNdiag=False
MNinv=False

# ROOTPATH = 'C:/Users/Guest_admin/Downloads/GUMData/'
ROOTPATH = 'D:/CACD2000.tar/CACD2000/CACD2000/'
train_txt = 'trainingAnnotations.txt'
val_txt = 'validationAnnotations.txt'
test_txt = 'testAnnotations.txt'

FC = 4096 # 4096
alpha = 0.75
N_Class = 3
LOW_DIM = 1
idOar = Age
argvs = ['-i', '-u']
ssRatio = 1.0  # float(sys.argv[3])/100.0
PB_FLAG = "Shared"  # to modify according to the task. A different evaluation function (test.py) will be used depending on the problem

print (PB_FLAG)

for idarg,arg in enumerate(argvs):
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
    elif arg == '-rnTra' or arg =='-rntra':
        VALMODE = rnTra
    elif arg == '-rnHard' or arg =='-rnhard':
        VALMODE = rnHard
    elif arg == '-reEqui'or arg =='-reequi':
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
# ITER = 6
WIDTH = 224
BATCH_SIZE = 64
# BATCH_SIZE = 128
NB_EPOCH = 15
PATIENCE = 1
NB_EPOCH_MAX = 50
LEARNING_RATE = 1e-04
# LEARNING_RATE = 1e-03
# validationRatio = 0.80
validationRatio = 1.0
fileWInit = ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"


class MixtureModel:
    ''' Class of forward model'''

    def __init__(self):
        # Regression CNN
        model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        x = Flatten(name='flatten')(model_vgg16.output)
        
        r = Dense(FC, activation='relu', name='fc1')(x)
        r = Dense(FC, activation='relu', name='fc2')(r)
        r = BatchNormalization(name='bm1')(r)
        reg = Dense(LOW_DIM, activation='linear', name='regression')(r)

        c = Dense(FC, activation='relu', name='fc1_1')(x)
        c = Dense(FC, activation='relu', name='fc2_2')(c)
        # x = BatchNormalization(name='bm1')(x)
        cla = Dense(N_Class, activation='softmax', name='classification')(c)
        model = Model(input=model_vgg16.input, output=[cla,reg])

        self.network = model

        self.priorInit=0.95
        

        if INDEP_MODE:
            self.logU=-np.log(224)*np.ones(LOW_DIM)
            self.piIn=self.priorInit*np.ones(LOW_DIM)
            self.rni=[]

        elif PAIR_MODE:
            self.logU=-2*np.log(224)*np.ones(LOW_DIM/2)
            self.piIn=self.priorInit*np.ones(LOW_DIM/2)
            self.rni=[]
        else:
            self.logU=-np.log(224)
            self.piIn=self.priorInit
            self.rni=[]

        self.lamb=np.ones(LOW_DIM)
        self.bestLoss=np.inf
        
    def fit(self, ROOTPATH, trainT, valT, test_txt,learning_rate=0.1, itmax=2,validation=validationRatio,subsampling=1.0):
        '''Trains the model for a fixed number of epochs and iterations.
           # Arguments
                X_train: input data, as a Numpy array or list of Numpy arrays
                    (if the model has multiple inputs).
                Y_train : labels, as a Numpy array.
                batch_size: integer. Number of samples per gradient update.
                learning_rate: float, learning rate
                nb_epoch: integer, the number of epochs to train the model.
                validation_split: float (0. < x < 1).
                    Fraction of the data to use as held-out validation data.
                validation_data: tuple (x_val, y_val) or tuple
                    (x_val, y_val, val_sample_weights) to be used as held-out
                    validation data. Will override validation_split.
                it: integer, number of iterations of the algorithm

                

            # Returns
                A `History` object. Its `History.history` attribute is
                a record of training loss values and metrics values
                at successive epochs, as well as validation loss values
                and validation metrics values (if applicable).
            '''
        
        self.fileW=ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"
        self.fileWInit=ROOTPATH+"Forward_Uni_init"+PB_FLAG+"_"+idOar+"_weights.hdf5"
        self.fileWbest=ROOTPATH+"Forward_Uni_best"+PB_FLAG+"_"+idOar+"_weights.hdf5"
        print ("Training Forward")

        
        '''Fine tune the network according to our custom loss function'''
        
        # layer_nb=16 # number of finetunned layer
        # # train only some layers
        # for layer in self.network.layers[:layer_nb]:
        #     layer.trainable = False
        # for layer in self.network.layers[layer_nb:]:
        #     layer.trainable = True
        # self.network.layers[-1].trainable = True

         # compile the model
        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        self.network.summary()
        
        self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': tf.losses.mean_squared_error},
            metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/1000])

        self.network.save_weights(self.fileWInit)
        self.network.save_weights(self.fileWbest)

        self.rni=np.ones((len(trainT),1),dtype=np.float)*np.ones((1,LOW_DIM))
        improve=True
        for it in range(itmax):
            if it == 0:
                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)

            else:
                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)
            if not improve:
                break

            
            (gen_training, N_train), (gen_test, N_test) = load_data_generator_List(ROOTPATH, trainT[:], test_txt)
            Ypred, Ytrue = extract_XY_generator_multiple(self.network, gen_training, N_train)
            Ntraining=int(validationRatio*N_train)
            
            # for iterEm in range(6):
            #     self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     if UNI:
            #         self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     self.E_step(Ypred,Ytrue)

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                N_test = 100
                self.evaluate(gen_test, N_test, WIDTH)


    # def custom_loss(y_true, y_pred):
    #     return -K.mean(K.sum(rn*y_true*K.log(y_pred), axis=-1))


    def custom_loss(self):
        def loss(y_true, y_pred):
            # return -K.mean(K.sum(self.networkRn.input[1]*y_true*K.log(y_pred), axis=-1))

            y_pred = y_pred + 1e-15
            return -K.sum(y_true*K.log(y_pred), axis=-1)
            # return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # L_reg = K.square(y_pred_0*p - y_true_0*p)
            # L_cla = -K.sum(p*y_true_1*K.log(y_pred_1), axis=-1)

            # return alpha*L_reg + (1-alpha)*L_cla
        return loss
     
    def M_step_network(self, ROOTPATH, trainT, valT, test_txt, learning_rate, nbEpoch=NB_EPOCH):
        
        
        checkpointer = ModelCheckpoint(filepath=self.fileW,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')

        
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)
        

        if MNdiag:
            wei=np.multiply(self.rni,self.lamb)
            lrcoeff=LOW_DIM*1.0/sum(self.lamb)
        elif MNinv:
            wei=np.multiply(self.rni,np.reciprocal(self.lamb[:]))
            lrcoeff=sum(self.lamb)/(1.0*LOW_DIM)
        elif MNiso:
            wei=self.rni[:,:]
            lrcoeff=1.0

        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': tf.losses.mean_squared_error},
            metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/1000])

        # self.network.compile(optimizer=sgd,loss=self.custom_loss())

        # Replicate rni LOW_DIM times
        # wei = np.repeat(wei, N_Class, axis=1)

        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei, valMode=VALMODE, validation=validationRatio, subsampling=ssRatio)

        history=self.network.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoch,
                                             verbose=1,
                                             callbacks=[checkpointer,early_stopping],
                                             validation_data=gen_val,
                                             nb_val_samples=N_val)
        print (history.history)

        if min(history.history['val_loss'])<self.bestLoss:
            self.bestLoss=min(history.history['val_loss'])
            self.network.load_weights(self.fileW)
            self.network.save_weights(self.fileWbest)
            return True
        else:
            self.network.load_weights(self.fileWbest)
            return False



    
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
    
        i=0
        Ypred=[]
        Cpred=[]
        Y=[]
        C=[]
        for x,y in generator:
            if i>=n_predict:
                break
            pred = self.network.predict_on_batch(x)
            Ypred.extend(pred[1])
            Y.extend(y[1])
            c_pred = np.argmax(pred[0], axis=1)
            Cpred.extend(c_pred)
            c = np.argmax(y[0], axis=1)
            C.extend(c)
            i+=len(y[1])
        
        return np.asarray(Ypred), np.asarray(Y), np.asarray(Cpred), np.asarray(C)

   
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
        
        Ypred, Y, Cpred, C = self.predict(generator, n_eval)
        with open(Age+"_objs.pkl", 'wb') as f:
            pickle.dump([Ypred, Y, Cpred, C], f)
        f.close()
        # print ("Y: " + str(Y))
        # print ("Ypred: " + str(Ypred))
        run_eval(Ypred, Y, l, pbFlag, Cpred, C)


def readT(rootpath, file_train):
    return open(rootpath+file_train, 'r', encoding=encode).readlines()


if __name__ == '__main__':

    forward_Model = MixtureModel()
    trainingT=readT(ROOTPATH,train_txt)
    trainingT = trainingT[:128]
    validationT=readT(ROOTPATH,val_txt)
    validationT = validationT[:128]
    forward_Model.fit(ROOTPATH, trainingT, validationT, test_txt,learning_rate=LEARNING_RATE,
                    itmax=ITER,validation=validationRatio,subsampling=ssRatio)
    (_,_), (gen_test, N_test) = load_data_generator(ROOTPATH, train_txt[:], test_txt,validation=1.0,subsampling=ssRatio)
    N_test = 100
    forward_Model.evaluate(gen_test, N_test, WIDTH)
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)

            