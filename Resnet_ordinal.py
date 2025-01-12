'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
import os
# added
import tensorflow as tf
from keras import backend as K

from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from VGG16_rn import extract_XY_generator, extract_XY_generator_multiple
from data_generator_resnet_ordinal import load_data_generator,load_data_generator_simple
from data_generator_resnet_ordinal import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
from scipy.special import logsumexp
from log_gauss_densities import gausspdf,loggausspdf
from test import run_eval

start_time = time.time()
encode = "ISO-8859-1"
Age = "test" # "C"
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

FC1 = 1024 # 4096
FC2 = 256
alpha = 0.75
N_Bin = 48
N_Class = 3 # 3
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
# ITER = 6
ITER = 2
WIDTH = 224
BATCH_SIZE = 64
# BATCH_SIZE = 128
NB_EPOCH = 15
PATIENCE = 1
NB_EPOCH_MAX = 50
LEARNING_RATE = 1e-04
# LEARNING_RATE = 1e-05
# validationRatio = 0.80
validationRatio = 1.0
fileWInit = ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"


class MixtureModel:
    ''' Class of forward model'''

    def __init__(self):
        # Regression CNN
        # model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        model_resnet50 = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
        # for layer in model_vgg16.layers[:15]:
        #     layer.trainable = False

        for layer in model_resnet50.layers[:54]:
            layer.trainable = False
        x = Flatten(name='flatten')(model_resnet50.output)
        
        r = Dense(FC1, activation='relu', name='fc1')(x)
        r = Dense(FC2, activation='relu', name='fc2')(r)

        # Dynamically add layers
        REG = []
        for i in range(N_Bin):
	        globals()["reg_" + str(i)] = Dense(1, activation='sigmoid', name='regression_'+str(i))(r)
            # REG.extend(globals()["reg_" + str(i)])

        # Short shared layers
        # s = model_resnet50.layers[112].output
        # y = Flatten(name='flatten2')(s)

        # r = Dense(FC1, activation='relu', name='fc1')(x)
        # r = Dense(FC2, activation='relu', name='fc2')(r)
        # reg = Dense(LOW_DIM, activation='linear', name='regression')(r)

        # r = BatchNormalization(name='bm1')(r)
        # reg = Dense(N_Bin, activation='softmax', name='regression')(x)
        # reg = Dense(N_Bin, activation='sigmoid', name='regression')(x)
        # reg = Dense(LOW_DIM, activation='linear', name='regression')(y)
        
        c = Dense(FC1, activation='relu', name='fc1_1')(x)
        c = Dense(FC2, activation='relu', name='fc2_2')(c)
        cla = Dense(N_Class, activation='softmax', name='classification')(c)

        # x = BatchNormalization(name='bm1')(x)
        # cla = Dense(N_Class, activation='softmax', name='classification')(x)
        # cla = Dense(N_Class, activation='softmax', name='classification')(y)
        model = Model(input=model_resnet50.input, output=[cla,reg_0,reg_1,reg_2,reg_3,reg_4,reg_5,reg_6,reg_7,reg_8,reg_9,reg_10,reg_11,reg_12,
        reg_13,reg_14,reg_15,reg_16,reg_17,reg_18,reg_19,reg_20,reg_21,reg_22,reg_23,reg_24,reg_25,reg_26,reg_27,reg_28,reg_29,reg_30,reg_31,reg_32,
        reg_33,reg_34,reg_35,reg_36,reg_37,reg_38,reg_39,reg_40,reg_41,reg_42,reg_43,reg_44,reg_45,reg_46,reg_47])

        # Alternative training
        # model.layers[20].trainable = False
        # model.layers[22].trainable = False
        # model.layers[24].trainable = False
        # model.layers[26].trainable = False

        # model.layers[21].trainable = False
        # model.layers[23].trainable = False
        # model.layers[25].trainable = False

        rn = Input(shape=(LOW_DIM,), name='input_2')
        # weightedReg = merge([rn,reg], mode='mul', name='merge')
        # modelRn = Model(input=[model_vgg16.input,rn], output=weightedRn)
        modelRn = Model(input=[model_resnet50.input,rn], output=[cla,reg_0,reg_1,reg_2,reg_3,reg_4,reg_5,reg_6,reg_7,reg_8,reg_9,reg_10,reg_11,reg_12,
        reg_13,reg_14,reg_15,reg_16,reg_17,reg_18,reg_19,reg_20,reg_21,reg_22,reg_23,reg_24,reg_25,reg_26,reg_27,reg_28,reg_29,reg_30,reg_31,reg_32,
        reg_33,reg_34,reg_35,reg_36,reg_37,reg_38,reg_39,reg_40,reg_41,reg_42,reg_43,reg_44,reg_45,reg_46,reg_47])

        self.network,self.networkRn = model,modelRn

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
        adam = Adam(lr=learning_rate)
        rms = RMSprop(lr=learning_rate)
        self.network.summary()
        self.networkRn.summary()
        
        # load previous trained weights
        # self.network.load_weights(self.fileWbest)

        # self.networkRn.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'merge': 'mse'},
        #     metrics={'classification': 'accuracy', 'merge': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/100])
        # self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': 'mse'},
        #     metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/100])

        # self.networkRn.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
        #     metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])
        # self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
        #     metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])
        
        self.networkRn.compile(optimizer=sgd, loss=[self.custom_loss(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2()], metrics=['accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy'], loss_weights=[1-alpha, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10])
        self.network.compile(optimizer=sgd, loss=[self.custom_loss(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2()], metrics=['accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy'], loss_weights=[1-alpha, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10])

        self.network.save_weights(self.fileWInit)
        self.network.save_weights(self.fileWbest)

        self.rni=np.ones((len(trainT),1),dtype=np.float)*np.ones((1,LOW_DIM))
        improve=True
        for it in range(itmax):
            if it == 0:
                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)

            else:
                # f = open('test2_rni_objs.pkl','rb')
                # self.rni = pickle.load(f)
                # f.close()

                # for i in range(self.rni.shape[0]):
                #     if self.rni[i] < 0.001:
                #         self.rni[i] = 0.001

                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)
            if not improve:
                break

            
            (gen_training, N_train), (gen_test, N_test) = load_data_generator_List(ROOTPATH, trainT[:], test_txt)
            Ypred, Ytrue = extract_XY_generator_multiple(self.network, gen_training, N_train)
            Ntraining=int(validationRatio*N_train)
            
            for iterEm in range(6):
                self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
                if UNI:
                    self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
                self.E_step(Ypred,Ytrue)

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                # N_test = 100
                self.evaluate(gen_test, N_test, WIDTH)


    # def custom_loss(y_true, y_pred):
    #     return -K.mean(K.sum(rn*y_true*K.log(y_pred), axis=-1))

    def custom_loss(self):
        def loss(y_true, y_pred):
            # return -K.mean(K.sum(self.networkRn.input[1]*y_true*K.log(y_pred), axis=-1))

            p = self.networkRn.input[1]
            # y_pred = y_pred + 1e-15
            return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # L_reg = K.square(y_pred_0*p - y_true_0*p)
            # L_cla = -K.sum(p*y_true_1*K.log(y_pred_1), axis=-1)

            # return alpha*L_reg + (1-alpha)*L_cla
        return loss

    def custom_loss2(self):
        def loss2(y_true, y_pred):
            # return -K.mean(K.sum(self.networkRn.input[1]*y_true*K.log(y_pred), axis=-1))
            # Bin = np.expand_dims(np.arange(14, 63), axis=1)
            p = self.networkRn.input[1]
            # L = K.square(y_pred@Bin*p - y_true@Bin*p)
            # return K.mean(L, axis=0)
            # y_pred = p*y_pred
            # y_pred = K.round(y_pred)
            # y_pred = K.sum(y_pred, axis=-1) + 14
            # y_true = K.sum(y_true, axis=-1) + 14
            # return K.mean(K.square(y_true - y_pred))


            return -K.sum(p*y_true*K.log(y_pred) + p*(1-y_true)*K.log(1-y_pred), axis=-1)
            # y_pred = y_pred + 1e-15
            # return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # L_reg = K.square(y_pred_0*p - y_true_0*p)
            # L_cla = -K.sum(p*y_true_1*K.log(y_pred_1), axis=-1)

            # return alpha*L_reg + (1-alpha)*L_cla
        return loss2


    def getRn(self,Ypred,Ytrue):
        if INDEP_MODE:
            rni=np.ones((Ypred.shape[0],1),dtype=np.float)*np.ones((1,LOW_DIM))
            for i in range(LOW_DIM):
                logrni = np.ndarray(Ytrue.shape[0])
                umat= np.ndarray(Ytrue.shape[0])
                lognormrni = np.ndarray(Ytrue.shape[0])
                logrni[:] = np.log(self.piIn[i])+loggausspdf(Ypred[:,i].reshape((Ypred.shape[0],1)),Ytrue[:,i].reshape((Ytrue.shape[0],1)), self.lamb[i])
                umat=(np.log(1-self.piIn[i])+self.logU[i])*np.ones(logrni.shape[0])
                # umat=(np.log(1-self.piIn[i]+0.001)+self.logU[i])*np.ones(logrni.shape[0])
                lognormrni = logsumexp(np.stack([logrni,umat]),axis=0)
                rni[:,i]=np.exp(logrni- lognormrni)
            return rni
        if PAIR_MODE:
            rni=np.ones((Ypred.shape[0],1),dtype=np.float)*np.ones((1,LOW_DIM))
            for i in range(LOW_DIM/2):
                logrni = np.ndarray(Ytrue.shape[0])
                umat= np.ndarray(Ytrue.shape[0])
                lognormrni = np.ndarray(Ytrue.shape[0])
                logrni[:] = np.log(self.piIn[i])+loggausspdf(Ypred[:,2*i].reshape((Ypred.shape[0],1)),Ytrue[:,2*i].reshape((Ytrue.shape[0],1)), self.lamb[2*i])+loggausspdf(Ypred[:,2*i+1].reshape((Ypred.shape[0],1)),Ytrue[:,2*i+1].reshape((Ytrue.shape[0],1)), self.lamb[2*i+1])
                umat=(np.log(1-self.piIn[i])+self.logU[i])*np.ones(logrni.shape[0])
                lognormrni = logsumexp(np.stack([logrni,umat]),axis=0)
                rni[:,2*i]=np.exp(logrni- lognormrni)
                rni[:,2*i+1]=rni[:,2*i]
            return rni
        
        else:
            logrni = np.ndarray(len(Ytrue))

            logrniI = np.ndarray((len(Ytrue),LOW_DIM))
            for i in range(LOW_DIM):
                logrniI[:,i]=loggausspdf(Ypred[:,i].reshape((Ypred.shape[0],1)),Ytrue[:,i].reshape((Ytrue.shape[0],1)), float(self.lamb[i]))
            logrni[:] =np.sum(logrniI,axis=1)+np.log(self.piIn)
            umat=(np.log(1-self.piIn)+self.logU*LOW_DIM)*np.ones(logrni.shape[0])
            lognormrni = logsumexp(np.stack([logrni[:],umat]),axis=0)
            rnik=np.exp(logrni[:]- lognormrni[:])
            return rnik.reshape(rnik.shape[0],1)*(np.ones((1,LOW_DIM)))
            
   
            

    def E_step(self,Ypred,Ytrue):
        self.rni=self.getRn(Ypred,Ytrue)
        with open(Age+"_rni_objs.pkl", 'wb') as f:
            pickle.dump(self.rni, f)
        f.close()
        print ("rni mean: " + str(np.sum(self.rni,axis=0)/(self.rni.shape[0])))
            
    def M_step_lambda(self,Ypred,Ytrue):

        lamb=np.empty(LOW_DIM)
        for i in range(LOW_DIM):
            diffSigmakList = np.sqrt(self.rni[:Ypred.shape[0],i]).T*(Ypred[:,i]-Ytrue[:,i]).T
            lamb[i]=np.sum(diffSigmakList**2)/(np.sum(self.rni[:Ypred.shape[0],i]))

        if DIAG:

            self.lamb=lamb
        elif ISO:

            self.lamb=np.sum(lamb)/LOW_DIM*np.ones(LOW_DIM)
        elif PAIR_MODE:

            for i in range(LOW_DIM/2):
                diffSigmakList = np.sqrt(self.rni[:Ypred.shape[0],2*i:2*(i+1)]).T*(Ypred[:,2*i:2*(i+1)]-Ytrue[:,2*i:2*(i+1)]).T
                self.lamb[2*i]=np.sum(diffSigmakList**2)/(np.sum(self.rni[:Ypred.shape[0],2*i:2*(i+1)]))
                self.lamb[2*i+1]=self.lamb[2*i]
        print ("lambda: " + str(self.lamb))

    def M_step_U(self,Ypred,Ytrue):
        err=Ypred-Ytrue
        if U_MIN_MAX==True:
            # this implementation differs from the equations in the papers. Here, we simply use the min and max of the error. It turned out to be simpler, faster and performs similarly.

            if INDEP_MODE:
                for i in range(LOW_DIM):
                    self.logU[i]=-np.log(np.max(err[:,i])-np.min(err[:,i]))
            elif PAIR_MODE:
                for i in range(LOW_DIM/2):
                    self.logU[i]=-np.log(np.max(err[:,2*i])-np.min(err[:,2*i]))-np.log(np.max(err[:,2*i+1])-np.min(err[:,2*i+1]))

        else:
            # ri = (LOW_DIM*Ypred.shape[0])-(np.sum(self.rni[:Ypred.shape[0],:]))/(LOW_DIM*Ypred.shape[0])
            ri = LOW_DIM*Ypred.shape[0]-np.sum(self.rni[:Ypred.shape[0],:])
            mu1 = np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*(Ypred[:,:]-Ytrue[:,:]))/ri
            mu2 = np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*((Ypred[:,:]-Ytrue[:,:]))**2)/ri
            self.logU=-np.log(2*np.sqrt(3*(mu2-mu1**2)))
            # print "U: " + str(self.logU)
            # self.logU=np.sum(-np.log(np.max(err,axis=0)-np.min(err,axis=0)))/LOW_DIM
        print ("U: " + str(self.logU))

            
    def M_step_pi(self,Ypred,Ytrue):
        if INDEP_MODE:
            for i in range(LOW_DIM):
                self.piIn[i]=np.sum(self.rni[:Ypred.shape[0],:],axis=0)/(Ytrue.shape[0])
        elif PAIR_MODE:
            piCp=np.sum(self.rni[:Ypred.shape[0],:],axis=0)/(Ytrue.shape[0])
            for i in range(LOW_DIM/2):
                self.piIn[i]=piCp[2*i]
        else:
            self.piIn=np.sum(self.rni[:Ypred.shape[0],:])/len(Ytrue)
        print ("piIn: " + str(self.piIn))
        
    def M_step_network(self, ROOTPATH, trainT, valT, test_txt, learning_rate, nbEpoch=NB_EPOCH):
        
        
        checkpointer = ModelCheckpoint(filepath=self.fileW,
                                       monitor='val_loss',
                                       verbose=2,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')

        
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=2)
        

        if MNdiag:
            wei=np.multiply(self.rni,self.lamb)
            lrcoeff=LOW_DIM*1.0/sum(self.lamb)
        elif MNinv:
            wei=np.multiply(self.rni,np.reciprocal(self.lamb[:]))
            lrcoeff=sum(self.lamb)/(1.0*LOW_DIM)
        elif MNiso:
            wei=self.rni[:,:]
            lrcoeff=1.0

        sgd = SGD(lr=learning_rate*lrcoeff,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)
        adam = Adam(lr=learning_rate)
        rms = RMSprop(lr=learning_rate)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        # self.networkRn.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'merge': 'mse'},
        #     metrics={'classification': 'accuracy', 'merge': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/100])
        # self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': 'mse'},
        #     metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/100])

        # self.networkRn.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
        #     metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])
        # self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
        #     metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])

        self.networkRn.compile(optimizer=sgd, loss=[self.custom_loss(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2()], metrics=['accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy'], loss_weights=[1-alpha, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10])
        self.network.compile(optimizer=sgd, loss=[self.custom_loss(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),self.custom_loss2(),
        self.custom_loss2(),self.custom_loss2(),self.custom_loss2()], metrics=['accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy',
        'accuracy','accuracy','accuracy'], loss_weights=[1-alpha, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, 
        alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10, alpha/10])

        # self.network.compile(optimizer=sgd,loss=self.custom_loss())

        # Replicate rni LOW_DIM times
        # wei = np.repeat(wei, N_Class, axis=1)

        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei, valMode=VALMODE, validation=validationRatio, subsampling=ssRatio)

        history=self.networkRn.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoch,
                                             verbose=2,
                                             callbacks=[checkpointer,early_stopping],
                                             validation_data=gen_val,
                                             nb_val_samples=N_val)
        print (history.history)

        if min(history.history['val_loss'])<self.bestLoss:
            self.bestLoss=min(history.history['val_loss'])
            self.networkRn.load_weights(self.fileW)
            self.networkRn.save_weights(self.fileWbest)
            return True
        else:
            self.networkRn.load_weights(self.fileWbest)
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
        Bin = np.expand_dims(np.arange(14, 63), axis=1)
        i=0
        Ypred=[]
        Cpred=[]
        Y=[]
        C=[]
        for x,y in generator:
            if i>=n_predict:
                break
            pred = self.network.predict_on_batch(x)
            tmp = np.zeros(pred[1].shape)
            tmp2 = np.zeros(y[1].shape)
            for j in range(48):
                tmp = tmp + np.round(pred[j+1])
                tmp2 = tmp2 + np.round(y[j+1])

            pred_new = np.sum(tmp, axis=-1) + 14
            # pred_new = pred[1]@Bin
            Ypred.extend(np.squeeze(pred_new))
            # Ypred.extend(pred[1])
            y_new = tmp2 + 14
            # y_new = y[1]@Bin
            Y.extend(y_new)
            # Y.extend(y[1])
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
    # trainingT = trainingT[:128]
    validationT=readT(ROOTPATH,val_txt)
    # validationT = validationT[:128]
    forward_Model.fit(ROOTPATH, trainingT, validationT, test_txt,learning_rate=LEARNING_RATE,
                    itmax=ITER,validation=validationRatio,subsampling=ssRatio)
    (_,_), (gen_test, N_test) = load_data_generator(ROOTPATH, train_txt[:], test_txt,validation=1.0,subsampling=ssRatio)
    # N_test = 100
    forward_Model.evaluate(gen_test, N_test, WIDTH)
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)

            
