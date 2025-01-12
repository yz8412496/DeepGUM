'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
# import os
# import tensorflow as tf
# from keras import backend as K

from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
# from VGG16_rn import VGG16, extract_XY_generator
from VGG16_rn import extract_XY_generator
from data_generator import load_data_generator,load_data_generator_simple
from data_generator import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
from scipy.special import logsumexp
from log_gauss_densities import gausspdf,loggausspdf
from test import run_eval


# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.80
# sess = tf.Session(config=config)

start_time = time.time()
encode = "ISO-8859-1"
Age = "test3" # 'Y', 'M', 'O' and 'All'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DISPLAY_TEST=True
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
LOW_DIM = 1
idOar = Age
argvs = ['-i', '-u']
# ROOTPATH=sys.argv[1]
# train_txt = sys.argv[2]
# test_txt = sys.argv[3]
# LOW_DIM = int(sys.argv[4])
ssRatio = 1.0  # float(sys.argv[3])/100.0
# idOar=sys.argv[5]
PB_FLAG = "PROBLEM"  # to modify according to the task. A different evaluation function (test.py) will be used depending on the problem

print (PB_FLAG)

# for idarg,arg in enumerate(sys.argv):
for idarg,arg in enumerate(argvs):
    if arg == '-u':
        UNI = True
    elif arg == '-i':
        INDEP_MODE = True
    elif arg == '-p':
        DIAG = False
        PAIR_MODE = True
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
ITER = 1
# ITER = 6
WIDTH = 224
BATCH_SIZE = 64
# BATCH_SIZE = 128
# NB_EPOCH = 15
NB_EPOCH = 50
PATIENCE = 1
NB_EPOCH_MAX = 50
LEARNING_RATE = 1e-04
# validationRatio = 0.80
validationRatio = 1.0
fileWInit = ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"


class MixtureModel:
    ''' Class of forward model'''

    def __init__(self):
        '''Initialize VGG16 model'''
        # model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        # for layer in model_vgg16.layers[:15]:
        #     layer.trainable = False
        model_resnet50 = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')

        for layer in model_resnet50.layers[:54]:
            layer.trainable = False
        x = Flatten(name='flatten')(model_resnet50.output)
        # x = Dense(4096, activation='relu', name='fc1')(x)
        # x = BatchNormalization(name='bm1')(x)
        # x = Dense(4096, activation='relu', name='fc2')(x)
        # x = BatchNormalization(name='bm1')(x)
        x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
        model = Model(model_resnet50.input, x)

        rn = Input(shape=(LOW_DIM,))
        weightedRn = merge([rn,x],mode='mul')
        modelRn = Model(input=[model_resnet50.input,rn], output=weightedRn)

        # model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        # model_vgg16.trainable = False

        # model = Sequential([
        #     model_vgg16,
        #     Flatten(name='flatten'),
        #     Dense(4096, activation='relu', name='fc1'),
        #     Dense(4096, activation='relu', name='fc2'),
        #     Dense(LOW_DIM, activation='linear', name='predictions')
        # ])

        # modelRn = Sequential([
        #     model_vgg16,
        #     Flatten(name='flatten'),
        #     Dense(4096, activation='relu', name='fc1'),
        #     Dense(4096, activation='relu', name='fc2'),
        #     rn = Input(shape=(LOW_DIM,))
        #     Dense(LOW_DIM, activation='linear', name='predictions')
        # ])

        # self.network,self.networkRn = VGG16(LOW_DIM,weights='imagenet')
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
        # for layer in self.networkRn.layers[:layer_nb]:
        #     layer.trainable = False
        # for layer in self.networkRn.layers[layer_nb:]:
        #     layer.trainable = True
        # self.networkRn.layers[-1].trainable = True

        # compile the model
        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        # rmsprop = RMSprop(lr=learning_rate, clipnorm=1.0)

        self.network.summary()
        self.networkRn.summary()

        
        self.networkRn.compile(optimizer=sgd,
                               loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

        self.network.save_weights(self.fileWInit)
        self.network.save_weights(self.fileWbest)
        
        # self.networkRn.save_weights(self.fileWInit)
        # self.networkRn.save_weights(self.fileWbest)

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
            Ypred, Ytrue = extract_XY_generator(self.network, gen_training, N_train)
            Ntraining=int(validationRatio*N_train)
            
            for iterEm in range(6):
                self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
                if UNI:
                    self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
                self.E_step(Ypred,Ytrue)

            with open(Age+"_rni_objs.pkl", 'wb') as f:
                pickle.dump(self.rni, f)
            f.close()

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                # N_test = 100
                self.evaluate(gen_test, N_test, WIDTH)
        

    
    def getRn(self,Ypred,Ytrue):
        if INDEP_MODE:
            rni=np.ones((Ypred.shape[0],1),dtype=np.float)*np.ones((1,LOW_DIM))
            for i in range(LOW_DIM):
                logrni = np.ndarray(Ytrue.shape[0])
                umat= np.ndarray(Ytrue.shape[0])
                lognormrni = np.ndarray(Ytrue.shape[0])
                logrni[:] = np.log(self.piIn[i])+loggausspdf(Ypred[:,i].reshape((Ypred.shape[0],1)),Ytrue[:,i].reshape((Ytrue.shape[0],1)), self.lamb[i])
                umat=(np.log(1-self.piIn[i])+self.logU[i])*np.ones(logrni.shape[0])
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
                    print ("U: " + str(self.logU[i]))
            elif PAIR_MODE:
                for i in range(LOW_DIM/2):
                    self.logU[i]=-np.log(np.max(err[:,2*i])-np.min(err[:,2*i]))-np.log(np.max(err[:,2*i+1])-np.min(err[:,2*i+1]))
                    print ("U: " + str(self.logU[i]))

        else:
            ri=(LOW_DIM*Ypred.shape[0])-(np.sum(self.rni[:Ypred.shape[0],:]))/(LOW_DIM*Ypred.shape[0])
            mu1= np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*(Ypred[:,:]-Ytrue[:,:]))/ri
            mu2= np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*((Ypred[:,:]-Ytrue[:,:]))**2)/ri
            self.logU=-np.log(2*np.sqrt(3*mu2-mu1**2))
            # print "U: " + str(self.logU)
            # self.logU=np.sum(-np.log(np.max(err,axis=0)-np.min(err,axis=0)))/LOW_DIM

            
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
        
    def M_step_network(self, ROOTPATH,trainT, valT, test_txt, learning_rate,nbEpoch=NB_EPOCH):
        
        
        checkpointer = ModelCheckpoint(filepath=self.fileW,
                                       monitor='loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')

        
        early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=1)
        

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

        # rmsprop = RMSprop(lr=learning_rate, clipnorm=1.0)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.networkRn.compile(optimizer=sgd,
                               loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

        
        
        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei,valMode=VALMODE, validation=validationRatio,subsampling=ssRatio)

        history=self.networkRn.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoc�;�l�R�~_���]iQ=�~��ϱkː���R/�c�V'�%_��ĭl�RBw�Y[����>/���(�!~g��W��c�3�*#���=x�]s�Z�|�7�iL�
#��u:X��r,�b$�d����K�H�	G�M:Jl��)baW�K����[�����O����}AO���X�A<Ϯ+à�ýnS���z��V;�,���qm�t�X�ERL�J.H�v55(��qw��T� �����?�o��2>jo����"6fi0�� ���?�����<��C��+�����d�T� �?�o����8�m]FrMJt�ةeAϰ������C�3۪$������}S�B�q[�������2#�f �?�����z=�v3�* ����|��g�UL��/���v;�,�"�s�8��x�]JFH�	jy�3�j5���z;��2*s����t�X��[�Ħ,�"�v5(�!x�uVX�o���"+�`���f;�,���r9m4�h�QFC�{�܈�E5��1|�0���@����4�h�QXÅnӶ"	&y%$�d��0�k�Р���-m"RfBU@����������褱D����T����̳�
0���p�[τ�܀��7��r1k�Ј��F%$�d�Ԏ ��ԅ`���!a'��QDÌ�����O��x��K�H�	H�	}9m7�iRQC�N��،�Z�����{Ϝ�� �?ǯ�C�N:K�Ȃ)^a��QNC����T���o���7�i|�#��	E9�:2l�R0�k�P���(��yG�VzA϶+� �'�%o��Ģ,�bvq�4��qU������Q~C��۱d�����S��#�f3�*����9|�2w�Y@����<�nӺ"�z5��1y+� �g�c���8��IBII;�,�"&p������MZJD�̩j����?���s��<�����
(��}w�YW�\��?�/�c�8�-{�\�F6zi�6#�&%?��ĳ�2�up���d�Ԉ��G�`/�#�f.U#���7��}q[���\�=0�k���*:`���!zg��V �'�s��,���|�7�)H�	w�M5
h��}c�V�1_����o�S�7�iO�Ӹ�FzM�v(�!e'��P��Ȯ)C���L�J���x�P�C�=;�l��w�YJE��j=�s�:��;�l��b�_��vY;�,��6q)�4���AVO�߸��EjL��#��1u+���w��c�4�(��L��X�Ey�:&l�4�h�QLÊ.�v�9e-�p�[��<�.���	m9m2RjBP�C�0��Ԁ�����arW�ATπ������s��2$�d�ԋ����I`��1q+����Нc�V!'��O��쨲J���g�t�ػ�L����A}�{ל�V�=_�GÍnS���yC�{����5Y(�!l�5Rh�QnC���t�X��Ul��/�c�V9-?�o�S�2~j_����b2VjAϳ��0���p���Ԯ ���g��N ˧�q\ۆ$�$�d�����x�]FFM
zx��f&U% �������[�D����������YZE���> ����+��7ϩk�����.(�v�=U.@���4�(�a[���\��2vjY�3��20�k�л�̶*	 �'�%~dߔ�Ѕc��&!%'��D�̈�@����|�^���L�J��~=�wÙn���x��N&K���\�};�l��Rs�ZD���Z�|��'��L����y?�/�c�7�)K���]MJ}�yg�Vp�ﴳȊ)X�w��N%����\�F�4�h��P���-7�iFQ�~ߺ'��j4���F;�,�b�p�Ǵ�H�I^I�=Y.E#��54��qK����P���#��?�/���6:i,�"#�f<��;��z?���3�*7��G�{�\��-4�h�QI�>/�#��)u!�uH��ei�0��� �?�o�S�:l��9RmR~B_�Gۍd�T�������.6c�1?������3�5@.m#�fU2@�O���輱N����UZ@���� ����'��{�ܸ�E:L��2(�apכ�T���_��=t�X��J��)y!'�eI�0�+� ����n,Ӣ"f}p���T� �۟���x��RBq[������0��̀�����s�>t�J ���uq۵d�ԙ`���c��<�.��
}8�mg�UR@�O�K�L�
8��}r^ZG��\�F �?�o����<�n	�2*z`���!u'��Ut�د�C��8��@�O�K爵YH�	l�=2njS���r3�j����<�.c�V�;߬��n|Ӟ"�qE���8��P�C�N(ˡh��]S�Bv{��6�3�*;���zv\�%=$�d������z ���5u(��ew��P���,���{��v
Y8�-l�R6BiQ;����qO��Ԙ��G��c�V0�+��-h�QvC�;����uL��%h��t���]0�k��s�Z(��l��	Ry]>Fo��r$�d�Ԅ����a6W�A?�����4���W��Oߋ�صeH��`��1o����/�c�V6A)�;Ǭ�BN~K����Qe������+��w˙h�P��<��9C�{�\���4�h�\�:,�b9m1k�P�C��%k�Є�܆&%6d��0��؀�_���rZ~Dߌ��d�Ծ ����d�Ԃ �gוaP׃�^7��]AO��x��@�O�㸶I:I,�")&a%��D�������L�J ��������~�7��iq�����U} ���W��}o�Sׂ!^g��]P�C�2{�\���7�izQö.	#�&%:d�Բ �g��e`���wǙmU@�O�K�Ⱦ)O�Ǹ�MBJNHˉh�U3������|�w��L�
���z>\�3�*&`���x��L�J�z�v<�.%#���<��s��