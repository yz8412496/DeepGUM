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
from data_generator_shared_ordinal import load_data_generator,load_data_generator_simple
from data_generator_shared_ordinal import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
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

FC1 = 256 # 4096
FC2 = 64
alpha = 0.75
N_Bin = 48
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
ITER = 6
# ITER = 2
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

        # Dynamically add layers
        # REG = []
        # for i in range(3):
	    #     globals()["reg_" + str(i)] = Dense(2, activation='softmax', name='regression_'+str(i))(x)
            # REG.extend(globals()["reg_" + str(i)])

        # Short shared layers
        # s = model_resnet50.layers[112].output
        # y = Flatten(name='flatten2')(s)

        # r = Dense(FC1, activation='relu', name='fc1')(x)
        # r = Dense(FC2, activation='relu', name='fc2')(r)
        # reg = Dense(LOW_DIM, activation='linear', name='regression')(r)

        # r = BatchNormalization(name='bm1')(r)
        # reg = Dense(N_Bin, activation='softmax', name='regression')(x)
        reg = Dense(N_Bin, activation='sigmoid', name='regression')(x)
        # reg = Dense(LOW_DIM, activation='linear', name='regression')(y)
        
        # c = Dense(FC1, activation='relu', name='fc1_1')(x)
        # c = Dense(FC2, activation='relu', name='fc2_2')(c)
        # cla = Dense(N_Class, activation='softmax', name='classification')(c)

        # x = BatchNormalization(name='bm1')(x)
        cla = Dense(N_Class, activation='softmax', name='classification')(x)
        # cla = Dense(N_Class, activation='softmax', name='classification')(y)
        model = Model(input=model_resnet50.input, output=[cla,reg])

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
        modelRn = Model(input=[model_resnet50.input,rn], output=[cla,reg])

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

        self.networkRn.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
            metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])
        self.network.compile(optimizer=sgd, loss={'classification': self.custom_loss(), 'regression': self.custom_loss2()},
            metrics={'classification': 'accuracy', 'regression': 'accuracy'}, loss_weights=[1-alpha, alpha/10])

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
            # ri = (LOW_DIM*Ypred.shape[0])-(np.sum(se��{�\��m?�o�S�B6Ni�8��RB~N_��؝eVT� �����+���j>P���:'��B4�h��@����t�X�Y<�.,�6i=.s���0�kϐ����/�c�2q*[�ć�rZqۼ��������Ԋ ���u`���atט�UG��_�G��{�\���8�mJRH�InI�2*u ���ut�إeD�̠���o�S��?�o��R8�mnRS�BNw��X�P���.?����>2o�S��/�c�:,�:l�>ro�S�� �gÕnӳ�
6x�q6[��<��ñn���RzB\�F+� �g��`����mw�YREL�J?���s�4�踱MK�H��Ui �?���>=/�cÖ.#��
8��~2_�G��o�S��;�l��
x�]zF\�*} �g�uP���n4Ө�F��w��t���\��m:Rl�R.Bc�V�4����Qo���'�eJT���o��3�j(��c��9Q-�~_��}k�P���^#��E6L�
18�p�[�D���Z��,�"&ze�� �'�%jd�ԣ��7�)fa��K����N����X�EO���(��D���Z�<��s�Z��,��6{��6�7�)o�ǲ-JbH�Ia	�1M+�`���a`ח�Qw��^��K�H��O��8��H�IJI�9i-"s�Z���>���̮*�����_��{�\��=NnK���RuX�E���'��t�؈�YD���<�n?����~6_��=o�SÂ.c��	Q9�>o�Sς+�`���QlÒ.c�V
A8ϭk�P�CÎ.����9Vm�_�G��qj[�ē�2jrP�C��4����_߇��ufX�`���!���W��r/�c�� �'ǥmD�L�JH�	~y�7�ia��J���}U@���[��|��״�H��]YE=�z3��0�+�����k�P��>%/��Ķ,�"&u%��t�؍eZT������~3����o����>o�S�B(�ak���Sӂ"���{�\�=:nlӒ"frU@�����@�Oϋ�ذ�K�Ȭ�B��̗�p�����;۬���|��W��J���ye�p��4�h��_��y6])a=�qC������И��F �'�ep��षĉl�2p�[�Ļ쌲
t�ؽeNTˀ���W�sߚ'��`��ȡiG�S�B�z+���7�)c��=K�H��^�=H�Is�4�(��Mw�YX�l��>"o�S�,�b?�o��6
i8�-c�VA=�{Ü��>����^*G��G�Mp�[�ıl���uRX�EnLӊ"�uu��ex��`�W�l��?�o�S�B9m;�l�RBrNZK�Ȝ�V?������0��� �������!|�5W��Ao����$�d�T� ����	y9-6biQ1�� �����􀸟�W�Apϛ�԰���حeBT�@�����Գ��7��eq۰��Ę��B�s�0��а���(�!Bg�U[�ğ��zs��$�$��������3�**`����yw�Vu�����g�{�܋��5H��qi�4���R�_�G�uZX��l��"~f_���c�V?�/ߣ��5m(�abW�AQ�����L��>(�sǚ-T�@�O��8�-K�H�I]	y=.vc�10�����̧�p���$���ě씲�s��%d�Դ����]eT� �ϟ���{ǜ�VA>O���خ%C������۷�t��5P���n;���~r_�G��`�W��o���:"l�R5h�Q����uK�ȕiP��6�;�,���~-�w�Ym|�^:G��B*N`˗�qS���p���T����7�iu�z��f=.p���0�+�`��ġl��RvBYE;��2�p���䨴�H��W�e?��г��6(�!q'��T������>7��C�/��̆* �g�}0�k琵S��)na��Js��d����{�\���<�nS��	^y�=VnA���t�؁e_��Эc�V.A#�?�o�S�B9m;�l�RBrNZK�Ȝ�V?������0��� �������!|�5W��Ao����$�d�T� ����	y9-6biQ1�� �����􀸟�W�Apϛ�԰���حeBT�@�����Գ��7��eq۰��Ę��B�s�0��а���(�!Bg�U[�ğ��zs��$�$��������3�**`����yw�Vu�����g�{�܋��5H��qi�4���R�_�G�uZX��l��"~f_���c�V?�/ߣ��5m(�abW�AQ�����L��>(�sǚ-T�@�O��8�-K�H�I]	y=.vc�10�����̧�p���$���ě씲�s��%d�Դ����]eT� �ϟ���{ǜ�VA>O���خ%C������۷�t��5P���n;���~r_�G��`�W��o���:"l�R5h�Q����uK�ȕiP��6�;�,���~-�w�Ym|�^:G��B*N`˗�qS���p���T����7�iu�z��f=.p���0�+�`��ġl��RvBYE;��2�p���䨴�H��W�e?��г��6(�!q'��T������>7��C�/��̆* �g�}0�k琵S��)na��Js��d����{�\���<�nS��	^y�=VnA���t�؁e_��Эc�V.A#���4��1^k���S�B!g��L����1zk�Ж#�&#�&�<����J�y|�&w�D���|�^<Ǯ-C�NK��y[��|���	Jy�9fmp�[�D�̎*����}V^A��[�D�������w�ju���z4��1E+���7��`�˱h��X��R�v.Y#�&�64�(�!K���Y\�,�">fo���#�f<�. ��y<�.&c��<������9L�
2x�]p�[��|�^ǵmH�IbII1	+� �'�ed�Ԡ����mfRU@�O���ؿ�O�����ANO��ؘ�UD�̯���;����>o�S�B(�ak���Sӂ"fw�P���=�_Ç�s�Z	�<�.c���:��:~lߒ'�ebT�@�ǻ�L�J
H��}iQ7��^��O�K눰�K���G�~z_���-a"W�AE���<�������|��g�H��{��6�8�-_�G�Mm
Rx�]nFS�~tߘ��E`���!p��T���o�S��4�h��^��JvH�	e9�0�k�P���n/���"}&^e��P�Cώ+۠��ĝl�Rs�Z��ܼ�;��2�t�؟�W��x��C�N!���M\�F(�!jg��S��7�is����yL�
&x�t�X�M<�n(ӡb�}QC��	[��<�nӰ��x�BvNY�8��V2A*O����qB[�D������к#��*5 ���u{�ܕf�3��7��{����:,�"8�muX�EzL��&(�!t瘵UH��o��2;�l���w�Yy<�n9�2j~P߃��5g��A`ϗ��p����0�k�P���^,Ǣ-FbMJq۹d��p��Ф�Ć,�"fq�����UN@ˏ��T�������t����Z���2�z0���0�+��w�o���)za׶!I'�%Y$�$��4�h��QS��w��IU	 �?�/�c��?�/��9=-.bc�V3��������
?���s�Z0��찲�x��AfO�����F>M/�c��%a$פ�D���ZD�����<��8��JH�I��7�)t���IH�	i9-3�jP��~?�����n7��R�_χ��p�[����>o���R)a>W��Cߎ'ۥd�Ԝ���=c�V�>����n)�2�}@�O��X��Ml�R(�anW��R�w�Yg�\��;�,�b
Vx�o�S�)>a/���F#�&e4�蠱Gˍh�QTÀ�÷�	s�4�h��N#���5\��1m+�`�W�AbO�K���F:zl��&"e&T� ��ȿ�O�����NK�Ȉ�YA���<�� �����L��h�Q|Þ<�nS�B
Nx˝h�QQ�����L�
8���_�G�z\��8�-BbNVK���W�j�����5vh�e3������(��N7��X�_���-k�P�C�&{����6):a,ע!Fg�Zp��줲�|��g��K�ȷ�Iq	�4�(�aT׀�_Ǉ�]rFZM�|��g��O����1I+� �'�%`����x��]VFA�{�ܪ& �?�����J>H�s�%4�贱H��X�U<��/��<�.>c���>#���>,�3�j-�s�Z9�<�n
S��nzS��.q#���0���0�+Ǡ�G�MnJS��nu��zp���$�$��t���S��=nnS��rw�YT� ���?�o��<�n:S��.~c���1c�� �?ӯ��~9�7�inQ��
w��Me
T���o�S�0�kא�Sǂ-^bG�MQ
C��k�P���.$㤶�<�.#���8��MrJZH��l�2�_����s�Z:D�̲*
`���a~W��W߁gߕg��c��7�)w�G�H�I|�)7�)G�G�ML�J(��iw�S��y��w�w�H�	x�}6^i�=S�B�~���ȑiS��r�{�ܠ��=l�R3�jP���^=�}C�N��X��X��V,�"/�c�,�";�l��v:Y,�",�b5h�{����5{�܁f�7��c�;�,���u]�um�ubX�Ea׺!L�5X��qlے$�d�T�@�O�K�H�II		99--"bfVU ������������� ��������̨�@����������T�@����䤴����V0����������0���0���������5s��d���Џ���$�$�$�d�Ԑ����-bbVVA���̫� ������O�����O���許AK����T� ��ܿ��;��rZ{�ܜ�1<�0��� ���O���(�aL׊!X�u\��%m$�d�T�@��{���	I9	-9"m&ReT�@�����䰴�Ș�UA Ͽ������=RnBS�B�t����P����"3�j���:?���3�j3����;�l��"{�\��6>i/�#Ӧ"&|�4���AM�{�ܥf�<���z\��<�.c��8��X�E^LǊ-X�EvL�
%8��t�X�EX��j,��#�f9-0�k�P��>*o����-vbYE1�0���0��İ����uC��k�Ћ�ض%I$�$�$�$���đl��rrZZD�̬� ����������W�~�����aw��QU��������7��|����Ii	93�*`�W��ߟ���axםaVW�_����k�P���)�Ƿ�IrII4�(�!A'��[�Ĭ��~w��W�`�������1}+�`���Qc��7��J����w��{����9M-
bx�]aW�N��ؗ�QtØ�C��븰�K�H��D��:���9nm�rZrD�L�����~����ai�1S�� �ן�W��y_��}aW��I_��=e.T〶�7�)q!���H��V)!?���C��2+�`����g��yP��~5���yo��r!g��H��G�q:[�Ă,�b�qQ�����QM�~ߵg��i`��v�8�-P�C�N9�8�m^RG�M^JG��YjE���:0���0�k�Ъ#��/�#��=u.X�v�6%)$�$���D��:���::l��2"jfP���7������z)�67�)A!���L��(�a���W�v�7�)`���yK��yQ�~	�7�)ja׳�J7��Yq��8��T�@�O׋�X��I\�)=!.g��F�3�j0����;�,�"&����rzt�ئ%E$��4���W��|��ױaK���YS��v?�/�#��7�)x�w�YI	<�.=#�f�>���:(��r7�iT� ����;�l�R�z.\�6)6a)�1G��@�O��਷�Io��2%*d�Է��w�e5��{˜��Q?����5O���h��Cӎ"�t���V8�-9RmR~B_�Gۍd�T�������.6c�1?������3�5@���k�����+�`�W��eo��Т#�f-"p�[����~_��ȭiBQC������;�젲�}h�Qg��^ǳ�J2H�Ip��4�(��X��]\�F-"zf\� �'�e��Ч��v,�"%&d������L�
���~_��̝jP���<��ô���^�}L�J'��Yt���B8�mk�P�C�N"K�H�	\�==.nc��2s�Z �����{�ܾ&�;�쬲
~xߝg�Ua ׿�O���X�N|˞(��QG��^G��H�I@��;�,���z�v9-5"h�Qu�������qHۉd��0����;�l�r~Z_��ܭfU>@����$��4����P����!g��EP���.0�� �?�/�c��;�,��	Fy:vl�%2d�T����س�J4��qA���Ȕ�P�߾'ϥk�Ь��.}#�f�1P��;׬�B�}[�D���Z���v�:%,��4�h�Y3�*��7�)}!g��IP��>1/����/�#�f5(��{���V:���'�es����ĺ,��6t��5K�ȁi_�ӽbV{����1i+� ���b|�^!��ENLˊ(��Uw��_���x�]O�K��yE�z9�62i*Q ç�s��$��������0�+� ����mx�]bFVM
���g�Up�����8��Ur@�O������M_�G��ejT�����3�*2`�W��{�
!8�uBX�Ek�К#�� �'��it���F�9jm�s�Z6D��:���=^nG��RBt�X��@���+� ���Ei�:#��5>h�sӚ"�p����x�_�G�i:Q,â.c�q;�����sך!T瀵_���mq[�D�L��h��{㜶	19+� �g�Ug��_����yr]Ft��u@���k�и��F*M �g��q`ۗ�t���Rp�[�D�����x��MfJU��o��r?�o���7�i8�mRRBBNNK����UU �������?���3��?������:<��23�j �����?�/�#�f?�/����69)-!"g�UE ���?������>�3��*0����{��v �?�/����=I.I#�&%5$��t����UP����3�: ���?�o���"?�o���2:jl��#�f6U) �?���C�>{����!7��EAϺ+��7��o���<�nӵb�ya�qI�4�(�