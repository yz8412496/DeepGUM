'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
import math
# import os
# import tensorflow as tf
# from keras import backend as K

from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, BatchNormalization, multiply, Dropout
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
Age = "test2" # 'Y', 'M', 'O' and 'All'
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
ITER = 2
# ITER = 6
WIDTH = 224
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
        '''Initialize VGG16 model'''
        model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        x = Flatten(name='flatten')(model_vgg16.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        # x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(name='bm1')(x)
        x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
        model = Model(model_vgg16.input, x)

        rn = Input(shape=(LOW_DIM,))
        # weightedRn = merge([rn,x],mode='mul')
        weightedRn = multiply([rn,x])
        modelRn = Model(inputs=[model_vgg16.input,rn], outputs=weightedRn)

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
        start_time_training = time.time()
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

        rmsprop = RMSprop(lr=learning_rate)

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
                  nesterov=True)

        rmsprop = RMSprop(lr=learning_rate)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.networkRn.compile(optimizer=sgd,
                               loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

        
        
        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei,valMode=VALMODE, validation=validationRatio,subsampling=ssRatio)

        # history=self.networkRn.fit_generator(gen_training,
        #                                      samples_per_epoch=N_train,
        #                                      nb_epoch=nbEpoch,
        #                                      verbose=1,
        #                                      callbacks=[checkpointer,early_stopping],
        #                                      validation_data=gen8��\�F
M8�mh�QbC�N���X�E|��*'��G��x�]@�O��x�]L�J-�yv]u=�us��d�Ի����	d��0�kې��Ē,�bVrAO��Ȩ�AA���̤�������˟��Q{����1K����_���w�Y����9u-�uvX�e<�� ���h��~#���1l�0�k�P�C�N-�x�]]F}zw��V%$���ď���p���$�䄴���Q5��������p���T� �?߯���n8ӭbV~A����d��0�����oǓ�R2BjNP˃�1W��@�����d���P��> ���z,��&6e)�0���@��;דּ�
.x�vY1+���7۩d������h��F�2jt�أ�F4�(�a@׏�[���\�}4�h��ES��.t㘶I0�+� �'˥h��\��=6ni�2�r �������}\�F'�%Zd�Ԭ���}g�UW��_���z&\�4�(�aO���X��F�6*i �'�v�<�.㼶	;�,�"ft���G��}j^Pǃ�^2G�M@�O���h��L���uzX��f,�" �g�x��{�\��9~m�w�YbEL�
;��rZt�ج�B�|�� ���O���5M(�ahבaS��^s��T�@��;�l���x�]II=	.y#�&e1배��ب�ADό�� ��Ŀ���t�؎%[�Ą���q7��T� �����k���S�*~`ߗ��uc��a0׫�@���[��<����^ǹmMJrH�Id��0�+ߠ���ml�R"BfNU����W�A�����t���UV@���̺*��7��~1����o��24�h��K㈶I5	(�!}'�eW��P����.'��F�<�n ӿ��{��6i;�,��r}^tǘ�UB@�O���L����q~[���ܑf�2�s��;�촲�yX�f|� ���E}�z'��V4�(��Cǎ-[�D�L�
x�{�\�=5.h�v�2*p�9D��z
\��=m.Rc�VA7��[�����+���G�e:T���/�c��1a+���GӍbVt���C��)k����J&H�	t��5Nhˑh��R�rZw��\��<�n����z\�>}/�cז!Q'��^Ǽ�NK�H��[��<����
	8�-}"^fG�P�C��>+�����-g�UF@��{�ܻ��:��r=ntӘ�Fp��t�؋�X��H��B)a;���B�r�t�ؐ�S��,�b�~���	h�}3�j��S��<�n��R�x�]W�A]�{��v<�.8�vY>E/���6$�$�$�����\���22jjP����63�* �������'�e|�� ���El��*"`�W�x���_���yN]�x�VvA�;��r��ȧ�Eqۺ$��4�萱S˂(�aW��Q_���g�UI �?�/�#��<�.�6	)9!-'�eFT� �������z8��f2U*@�����p��t����_����v2Y*E ���5p���d������� �g��j ����v8�-e"T�@����|����MY
E8��j2P�C��;묰��x��EVL�
/���v*Y �'��r4�h��D���4�h�M3�j��c��=a.W��F�7�id�����8�mEL�J:H��r)a4ר�AG��[�D�̄����7�)~a���Ic�15+���wߙg�`����|��	W�M?�o���b4�h�G��JH��x�A6O��8��C�N.K���]5h�~s����`��̑j���r<�n$Ӥ��|�w�K���Y}|��9W�B�_ۇ�t�X�S��.{���16k��3ߪ'��o����=FnM�r�ud�ԥ`��ܡf�=P�C�:��8�mW�ARO�K�H��EY�:,��26ji�3�6 �?�/����>/�c�)1!+���G��f*U ����s��=d�T������yt��uE��j8��c�V:A,Ϣ+�`��qn[����vY2E*L��7��qq=D�L��
��}x�]g�U] ���w�Y|�,��9Fmzr\�F$�$�d�ԟ����yc�q1������W�t����O����qM�t�ؕeP���7én���{�ܱf�8��S�B:Nl˒(�aRW�A^O���X�EE��*<��7�z��?�/�#�&;�,���v	95-(�avW�U?������5|��1g��@�����0�k��3�*/����-y"]&Fe�p���$�����ܗ�u3��p���伴���T� ������yl�&reT�����C�N0˫耱_ˇ�qV[�����k�����%zd�Ԧ �'��f4�(��G��yZ]�|�w�YO����=E.L�6�5q(ۡd���P�C�#���:��29*m �g�Uv@��;�츲JzH��f)!0��@���k�P����(��J��Y~E���9d��p�[�ĥl���vY=.|�6�1A+����ԭ`�W�Ag��[�ģ�2*v`��1t똰�K�ȣ�F1+�`���!d电P���n%���|��'��H��H�	A9�;�l�R�~_���]iQ=�~��ϱkː���R/�c�V'�%_��ĭl�RBw�Y[����>/���(�!~g��W��c�3�*#���=x�]s�Z�|�7�iL�
#��u:X��r,�b$�d����K�H�	G�M:Jl��)baW�K����[�����O����}AO���X�A<Ϯ+à�ýnS���z��V;�,���qm�t�X�ERL�J.H�v55(��qw��T� �����?�o��2>jo����"6fi0�� ���?�����<��C��+�����d�T� �?�o����8�m]FrMJt�ةeAϰ������C�3۪$������}S�B�q[�������2#�f �?�����z=�v3�* ����|��g�UL��/���v;�,�"�s�8��x�]JFH�	jy�3�j5���z;��2*s����t�X��[�Ħ,�"�v5<����
	8�-}"^fG�P�C��>+�����-g�UF@��{�ܻ��:��r=ntӘ�Fp��t�؋�X��H��B)a;���B�r�t�ؐ�S��,�b�~���	h�}3�j��S��<�n��R�x�]W�A]�{��v<�.8�vY>E/���6$�$�$�����\���22jjP����63�