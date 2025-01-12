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
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
start_time = time.time()
encode = "ISO-8859-1"
Age = "Y" # 'Y', 'M', 'O' and 'All'
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
train_txt1 = 'trainingYoung.txt'
train_txt2 = 'trainingMiddle.txt'
train_txt3 = 'trainingOld.txt'
val_txt = 'validationAnnotations.txt'
val_txt1 = 'validationYoung.txt'
val_txt2 = 'validationMiddle.txt'
val_txt3 = 'validationOld.txt'
test_txt = 'testAnnotations.txt'
test_txt1 = 'testingYoung.txt'
test_txt2 = 'testingMiddle.txt'
test_txt3 = 'testingOld.txt'

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
        # start_time_training = time.time()
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
        self.networkRn.summary()

        
        self.networkRn.compile(optimizer=sgd,
                               loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

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
            Ypred, Ytrue = extract_XY_generator(self.network, gen_training, N_train)
            Ntraining=int(validationRatio*N_train)
            
            for iterEm in range(6):
                self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
                if UNI:
                    self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
                self.E_step(Ypred,Ytrue)

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                self.evaluate(gen_test, N_test, WIDTH)

    

    def custom_loss(self,rn):
        def loss(y_true, y_pred):
            return K.mean(K.sum(rn*K.square(y_pred-y_true),axis=-1))
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
            # ri = (LOW_DIM*Ypred.shape[0])-(np.sum(self.rni[:Ypred.shape[0],:]))/(LOW_DIM*Ypred.shape[0])
            ri = LOW_DIM*Ypred.shape[0]-np.sum(self.rni[:Ypred.shape[0],:])
            mu1 = np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*(Ypred[:,:]-Ytrue[:,:]))/ri
            mu2 = np.sum((np.ones(self.rni[:Ypred.shape[0],:].shape)-self.rni[:Ypred.shape[0],:])*((Ypred[:,:]-Ytrue[:,:]))**2)/ri
            self.logU=-np.log(2*np.sqrt(3*(mu2-mu1**2)))
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

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.networkRn.compile(optimizer=sgd,
                               loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

        # self.network.compile(optimizer=sgd,loss=self.custom_loss(self.rni))

        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei,valMode=VALMODE, validation=validationRatio,subsampling=ssRatio)

        history=self.networkRn.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoch,
                                             verbose=1,
                                             callbacks=[checkpointer,early_stopping],
                                             validation_data=gen_val,
                                             nb_val_samples=N_val)
        print (history.history)
        # test
        # K.clear_session()
        # tf.reset_default_graph()

        if min(history.history['val_loss'])<self.bestLoss:
            self.bestLoss=min(history.history['val_loss'])
            self.networkRn.load_weights(self.fileW)
            self.networkRn.save_weights(self.fileWbest)
            return True
        else:
      �q��ԗ��wәbp������[����~����h�Ns����P���n*S���}s�Z��\���:l�;�l�R�unXӅb�v!'�%H��t��5_���mo�S�B"NfK���S�>~o����!bg�UQ ÿ�����<��<����	O��8�m@�O�K�H�	M9
m8�mbRVBAO��̘�@�������ϸ��@�O�����J;��rp�����D�L����{�ܣ�5=(�as��T����9p��t�X��Z,��,�b	y1+�`��1e+������b)a1��@��؛�T�����C�4����N��؉eY�0���0�kÐ�ò.
c��a:W��B/�cۖ$�$����|�^G�MJJH��ii3��
 ����_�����_���}~^_���]fFU ����'������z|��&'�%D�̴���_��}�_ׇ�]w�Y]|�>w��C� ����~߶'�%i$�$����<�n��

x��}f^U��_�G���_����x�]^FG�ZzD�̦* ���5���g�s��#��4�(��Yw�\�8�-~b_�G�c�V�:/���6.i#�&�2�|��縵MH�Ih�i3�*���}|�^'��]D�L�
x�]�_��}uX��y\�&}%d���P��~$ߤ�ąl��"q&[������;ˬ��^���W�Au���|���uNX˅h��V#�&�7��l��x�]X�Em�z"\�F5(�a|מ!W��E_���-d�T�@��;�,���yI	6y)!6g�A0ϫ�������1C�� ��ԏ���Թ`��qd۔����ܒ&e2T�@����谱Kˈ��AU���������'�p��ܤ��<��3�*����]~F_��}d�T���_���-u"X�Eu��%|��4���A\φ+� �g�l��;�l�R	y>]/�c�&q%�􄸜�VA4Ϩ��@�����$������~߳��5h��qc���}�_ׇ�]w�Y]|�>w��C� ����~߶'�%i$�$����<�n��

x��}f^U��_�G���_����x�]^FG�ZzD�̦* ���5���g�s��#��4�(��Yw�\�8�-~b_�G�c�V�:/���6.i#�&�2�|��縵MH�Ih�i3�*���}|�^'��]D�L�
x�]�_��}uX��y\�&}%d���P��~$ߤ�ąl��"q&[������;ˬ��^���W�Au���|���uNX˅h��V#�&�7��l��x�]X�Em�z"\�F5(�a|מ!W��E_���-d�T�@��;�,���yI	6y)!6g�A0ϫ�������1C�� ��ԏ���Թ`��qd۔����ܒ&e2T�@����谱Kˈ��AU���������'�p��ܤ��<��3�*����]~F_��}d�T���_���-u"X�Eu��%|��4���A\φ+� �g�l��;�l�R	y>]/�c�&q%�􄸜�VA4Ϩ��@�����$������~߳��5h��qc���0��� ��_��}z^\ǆ-]"FfM
p���d�T����ԧ��w��f%$��􌸚T�@���+렰���h�QE��?��ȳ�J1�p��t�؄�\���6i>Q/���'�%Nd˔���S߂'�eg��P����-s�ZD��z���<�.��FM;�l��bp�[����N�8��YrEL��8��ArO�K�Ƞ�G�o�S��*.`��y3�*`���|����QHÉn�2�yp��t���Yx�l�R9m>Ro�S�B'�e[�Đ���.rc�V�0�����/ã�3�*`����~���qhۑd����s�Z"D�L�
��}}^w��]U@��{�����;���Bz{�ܖ&%3��������O���h�K���X�x��~&_���l�R�{�\��]<�n-�rZ}�|��W��N/��؆%]$�d��p�[ׄ�\��>}/�cז!Q'��^Ǽ�NK�H��[��<����
	8�-}"^fG�P�C��>+�����-g�UF@��{�ܻ��:��r=ntӘ�Fp��t�؋�X��H��B)a;���B�r�t�ؐ�S��,�b�~���	h�}3�j��S��<�n��R�x�]W�A]�{��v<�.8�vY>E/���6$�$�$�����\���22jjP����63�* �������'�e|�� ���El��*"`�W�x���_���yN]�x�VvA�;��r��ȧ�Eqۺ$��4�萱S˂(�aW��Q_���g�UI �?�/�#��<�.�6	)9!-'�eFT� �������z8��f2U*@�����p��t����_����v2Y*E ���5p���d������� �g��j ����v8�-e"T�@����|����MY
E8��j2P�C��;묰��x��EVL�
/���v*Y �'��r4�h��D���4�h�M3�j��c��=a.W��F�7�id�����8�mEL�J:H��r)a4ר�AG��[�D�̄����7�)~a���Ic�15+���wߙg�`����|��	W�M?�o���b4�h�G��JH��x�A6O��8��C�N.K���]5h�~s����`��̑j���r<�n$Ӥ��|�w�K���Y}|��9W�B�_ۇ�t�X�S��.{���16k��3ߪ'��o����=FnM�r�ud�ԥ`��ܡf�=P�C�:��8�mW�ARO�K�H��EY�:,��26ji�3�6 �?�/����>/�c�)1!+���G��f*U ����s��=d�T������yt��uE��j8��c�V:A,Ϣ+�`��qn[����vY2E*L��7��qq������S� �g�������5y(�!fg�P����<��
���_���-p�[�D��:l��8�mFRMJ~H߉g�e0��ష��h�A3��������^>G��C�N$ˤ���\���|��״�H��]YE=�z3��0�+�����k�P��>%/��Ķ,�"&u%��t�؍eZT������~3����o����>o�S�B(�ak���Sӂ"fw�P���=�_Ç�s�Z	�<�.c���:��:~lߒ'�ebT�@�ǻ�L�J
H��}iQ7��^��O�K눰�K���G�~z_���-a"W�AE���<�������|��g�H��{��6�8�-_�G�Mm
Rx�]nFS�~tߘ��E`���!p��T���o�S��4�h��^��JvH�	e9�0�k�P���n/���"}&^e��P�Cώ+۠��ĝl�Rs�Z��ܼ�;��2�t�؟�W��x��C�N!���M\�F(�!jg��S��7�is����yL�
&x�t�X�M<�n(ӡb�}QC��	[��<�nӰ��x�BvNY�8��V2A*O����qB[�D������к#��*5 ���u{�ܕf�3��7��{����:,�"8�muX�EzL��&(�!t瘵UH��o��2;�l���w�Yy<�n9�2j~P߃��5g��A`ϗ��p����0�k�P���^,Ǣ-FbMJq۹d��p��Ф�Ć,�"fq�����UN@ˏ��T�������t����Z���2�z0���0�+��w�o���)za׶!I'�%Y$�$��4�h��QS��w��IU	 �?�/�c��?�/��9=-.bc�V3��������
?���s�Z0��찲�x��AfO�����F>M/�c��%a$פ�D���ZD�����<��8��JH�I��7�)t���IH�	i9-3�jP��~?�����n7��R�_χ��p�[����>o���R)a>W��Cߎ'ۥd�Ԝ���=c�V�>����n)�2�}@�O��X��Ml�R(