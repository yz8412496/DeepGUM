'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
import os
# added
import tensorflow as tf
from keras import backend as K

from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.vgg16 import VGG16
from VGG16_rn import extract_XY_generator, extract_XY_generator_multiple
from data_generator_regression_class import load_data_generator,load_data_generator_simple
from data_generator_regression_class import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
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

FC = 4096 # 4096
alpha = 0.75
N_Class = 3
LOW_DIM = 1
idOar = Age
argvs = ['-i', '-u']
ssRatio = 1.0  # float(sys.argv[3])/100.0
PB_FLAG = "Joint"  # to modify according to the task. A different evaluation function (test.py) will be used depending on the problem

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
WIDTH = 112
BATCH_SIZE = 64
# BATCH_SIZE = 128
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
        # Regression CNN
        model_vgg16 = VGG16(input_shape=(112,112,3), include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        # Classification CNN
        model_vgg16_2 = VGG16(input_shape=(112,112,3), include_top=False, weights='imagenet')
        for layer in model_vgg16_2.layers[:15]:
            layer.trainable = False
        for i, layer in enumerate(model_vgg16_2.layers[1:]):
            layer.name = layer.name + '_2'

        x = Flatten(name='flatten')(model_vgg16.output)
        x2 = Flatten(name='flatten2')(model_vgg16_2.output)
        # c = Concatenate(axis=1)([x,x2])

        c = merge([x, x2], mode='concat', concat_axis=-1, name='merge_1')
        c = Dense(FC, activation='relu', name='fc1')(c)
        c = Dense(FC, activation='relu', name='fc2')(c)
        c = BatchNormalization(name='bm1')(c)
        reg = Dense(LOW_DIM, activation='linear', name='regression')(c)

        x = Dense(FC, activation='relu', name='fc1_1')(x)
        x = Dense(FC, activation='relu', name='fc2_2')(x)
        # x = BatchNormalization(name='bm1')(x)
        cla = Dense(N_Class, activation='softmax', name='classification')(x)
        model = Model(input=[model_vgg16.input,model_vgg16_2.input], output=[cla,reg])

        rn = Input(shape=(LOW_DIM,), name='input_3')
        weightedReg = merge([rn,reg], mode='mul', name='merge_2')
        # modelRn = Model(input=[model_vgg16.input,rn], output=weightedRn)
        modelRn = Model(input=[model_vgg16.input,model_vgg16_2.input,rn], output=[cla,weightedReg])

        self.network,self.networkRn = model,modelRn

        self.priorInit=0.95
        

        if INDEP_MODE:
            self.logU=-np.log(112)*np.ones(LOW_DIM)
            self.piIn=self.priorInit*np.ones(LOW_DIM)
            self.rni=[]

        elif PAIR_MODE:
            self.logU=-2*np.log(112)*np.ones(LOW_DIM/2)
            self.piIn=self.priorInit*np.ones(LOW_DIM/2)
            self.rni=[]
        else:
            self.logU=-np.log(112)
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
                  nesterov=True,
                  clipnorm=1.0)
        rmsprop = RMSprop(lr=learning_rate, clipnorm=1.0)

        self.network.summary()
        self.networkRn.summary()

        
        self.networkRn.compile(optimizer=rmsprop, loss={'classification': self.custom_loss(), 'merge_2': tf.losses.mean_squared_error},
            metrics={'classification': 'accuracy', 'merge_2': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/10000])
        self.network.compile(optimizer=rmsprop, loss={'classification': self.custom_loss(), 'regression': tf.losses.mean_squared_error},
            metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/10000])

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
            
            # for iterEm in range(6):
            #     self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     if UNI:
            #         self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     self.E_step(Ypred,Ytrue)

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                # N_test = 100
                self.evaluate(gen_test, N_test, WIDTH)


    # def custom_loss(y_true, y_pred):
    #     return -K.mean(K.sum(rn*y_true*K.log(y_pred), axis=-1))


    def custom_loss(self):
        def loss(y_true, y_pred):
            # return -K.mean(K.sum(self.networkRn.input[1]*y_true*K.log(y_pred), axis=-1))

            p = self.networkRn.input[2]
            y_pred = y_pred + 1e-15
            return -K.sum(p*y_true*K.log(y_pred), axis=-1)
            # L_reg = K.square(y_pred_0*p - y_true_0*p)
            # L_cla = -K.sum(p*y_true_1*K.log(y_pred_1), axis=-1)

            # return alpha*L_reg + (1-alpha)*L_cla
        return loss


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

        sgd = SGD(lr=learning_rate*lrcoeff,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True,
                  clipnorm=1.0)
        rmsprop = RMSprop(lr=learning_rate, clipnorm=1.0)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.networkRn.compile(optimizer=rmsprop, loss={'classification': self.custom_lossuY�5l��1bk�P�Ӿ"�{���9>m/�c�V"A&O�����BN{����Q0ë� ������|����L�:��r8�md�T�@�O��x�H�Iy	96m)a2W�A@Ϗ���Č���p���ԫ�����g�q0۫䀴�ȗ�Qq������Ԩ��Gߍg�Ud�ԯ����9g�Bp�[넰����5G��AjO����1Fk��s�� ��ĵl��buX�{�܎&�4�蜱V�8��W�AnO����uFX�j|��#�5E(��j7��S�;�l���t�X�E]�z-�v6Y)!<�5C��k������$�d��0�+����9m-brVZAϼ�� ��������4��1T뀰����qG��T�@��Ы���/�#�&1%+����܉f50���p��̔�����'��~4ߨ��Eo���"$�d���{����=}.^c��Q6C�;�����t���X��{�ܲ&
e8��`�W�AhϑkӐ��r-bt�X�G��N*K�ȇ�]q[��|�����Kӈ�Fu�u|��%g��D�̇�p�[��<�n����yR]F~M�w��ee���̭jP�C��?ۯ���8��QBC�N�����YP���2?�o����3�j9�3�j:P���.:c��!>g��C��#�0�+��7�)l�7�iJQùn�r�z$��4�(��V7�)_�ǽmNRK�H�IW�Y?�/���65)(�!w��EU��/���6?�/�#�3�*,��7�iy3�j	�3�*>`���z#��51(�p���T�@����<�.�v:u,��%vd��0��ܰ��8��B2NjK�ȓ�R1k�P����$���D�̖* ��|��>'��C��,�� ���w�x�x�]y]=n}�r�qTۀ��ė�r�r�p��Ĥ����v�1U+������a}�qW��T�����g�v�3�*4���I{��6)0�+���G�jzP���.5#��u?���s��7��t��5P���n;���~r_�G��`�W��o���:"l�R5h�Q����uK�ȕiP��6�;�,���~-�w�Ym|�^:G��B*N`˗�qS���p���T����7�iu�z��f=.p���0�+�`��ġl��RvBYE;��2�p���䨴�H��W�e?��г��6(�!q'��T������>7��C�/��̆* �g�}0�k琵S��)na��Js��d����{�\���<�nS��	^y�=VnA���t�؁e_��Эc�V.A#���4��1^k���S�B!g��L����1zk�Ж#�&#�&�<����J�y|�&w�D���|�^<Ǯ-C�NK��y[��|���	Jy�9fmp�[�D�̎*����}V^A��[�D�������w�ju���z4��1E+���7��`�˱h��X��R�v.Y#�&�64�(�!K���Y\�,�">fo���#�f<�. ��y<�.&c��<������9L�
2x�]p�[��|�^ǵmH�IbII1	+� �'�ed�Ԡ����mfRU@�O���ؿ�O�����ANO��ؘ�UD�̯���;�����{��:;��2jw��S�4�h��OӋ��uI�5i(�!c��Q<î.���;�l�� ��_�G�My
]8�mmRrBZNDˌ��T��������p���d�T�������z7��V1+������d�T�@���k�з��v)!5'��AtϘ��@�����p�[�D�́j����b=nq���p���d�ԇ��w�Ya��N;����^pǛ�T�@�O���h��H��F5:h��r#�f�0����{�\�F�>o��Ȣ)Fa�qLۊ$��t�ؓ�R4�h�QC�����H�	U9 �?�o�S��>.o���-2bjVP��3Ϫ+�����n=�r�~߰���h��B#�f�4���R;�l�R�q^[���\�F3�j��#��>5/���v/�#�& �'��xv55(��qw��T� �����?�o��2>jo����"6fi0�� ���?�����<��C��+�����d�T� �?�o����8�m]FrMJt�ةeAϰ������C�3۪$������}S�B�q[�������2#�f �?�����z=�v3�* ����|��g�UL��/���v;�,�"�s�8��x�]JFH�	jy�3�j5���z;��2*s����t�X��[�Ħ,�"�v5(�!x�uVX�o���"+�`���f;�,���r9m4�h�QFC�{�܈�E5��1|�0���@����4�h�QXÅnӶ"	&y%$�d��0�k�Р���-m"RfBU@����������褱D����T����̳�
0���p�[τ�܀��7��r1k�Ј��F%$�d�Ԏ ��ԅ`���!a'��QDÌ�����O��x��K�H�	H�	}9m7�iRQC�N��،�Z�����{Ϝ�� �?ǯ�C�N:K�Ȃ)^a��QNC����T���o���7�i|�#��	E9�:2l�R0�k�P���(��yG�VzA϶+� �'�%o��Ģ,�bvq�4��qU������Q~C��۱d�����S��#�f3�*����9|�2w�Y@����<�nӺ"�z5��1y+� �g�c���8��IBII;�,�"&p������MZJD�̩j����?���s��<�����
(��}w�YW�\��?�/�c�8�-{�\�F6zi�6#�&%?��ĳ�2�up���d�Ԉ��G�`�W��~/����!m'�eRT�@�OË���J��i}s��	T� �?�o��;�l�RB}^{���VA1��������4����[܊&�5t�رeK�Ȑ�S�/�cϖ+� ���m<�n"S�B|��<��C��
��x�]T�@��{�\��
8�my]2FjM�s��1d따���آ%Fd��p�����x��V&A%��ļ���t����T�����s�Zw�e5��{˜��Q?����5O���h��Cӎ"�t���V8�-o�S�B-b{�\��2j{�ܓ�52h�QpÛ�������}D�L��X��~,ߢ'�em�p�[�D��z\��9]-b}^q��T�@��ԛ���Љc�%1$뤰��ܘ�E0���0�����L����%��ħ�r�v$�$�$�䜴��9S�~~_����afW�P����?��C��=k�P���'��ID��:,��?�o��r:Zl��,�bV}���W�q?��ԃ��7שaA��[˄���V�7ߩg�o����(�aI�1Y+� ���5a(סaG��QZC���� �?�/�#��8�-H�IvI	59(�!rg�UT�������:3��0�k������+�`��1jk�Г��6"i&Q%������봰���X�D�̾*�����~_�Gύk�P��Į,��w�Nu���}P�C�5[�āl���qb[�D����z:\��2-*b`�W�w��O�����I~I�7�)e!簵K�ȩiA�����|��?ׯ�C��9[��|�^��]K�H�	Vy?�o��21*k�Ї��v&Y%$��4����T�����'�uD�̥j����?�/�c�0�+Ӡ��}mRw�Y^E��Z>D�
$��t�X��T���/�#�6)<�.7��F?�o���"0�k����z*\��7�)raW��H��C�%;�섲�v�5e(��`���Qi�>���~(ߡgǕmP�C�N6K��9_��}n^S��^vG�U:@���+�`���!k���S��&.e#���3��)p����H�	Zy�<�n��:{�܂&e7��P�˾(��[Ǆ�\�FM7�iX�c��!;���B�v+� �'��c��8�-G�MFJM�yh�fs�����=L�J3��p����x�][�D��z�:	,�"=&ne���s��*$���x�U6@��;����{���V5?���s�3�� ���g�e0��ష��h�A3��������^>G��C�N$ˤ���\���1fk�����"<�n5��z���'�%w��D�����9;�,�bVw�_���i~Q���	g�M0�k�бc˖(�!S��^|Ǟ