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
Age = "All" # 'Y', 'M', 'O' and 'All'
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
WIDTH = 112
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
        model_vgg16 = VGG16(input_shape=(112,112,3), include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        x = Flatten(name='flatten')(model_vgg16.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bm1')(x)
        x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
        model = Model(model_vgg16.input, x)

        rn = Input(shape=(LOW_DIM,))
        weightedRn = merge([rn,x],mode='mul')
        modelRn = Model(input=[model_vgg16.input,rn], output=weightedRn)

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
                # N_test = 100
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
            self.networkRn.load_weights(self.fileWbest)
            return False



    
    def predict(self, generator, n_predict):
        '''Generates output predictions for the input samples,
           processing the samples in a batched way.
        # Arguments
     uzX��f,�" �g�x��{�\��9~m�w�YbEL�
;��rZt�ج�B�|�� ���O���5M(�ahבaS��^s��T�@��;�l���x�]II=	.y#�&e1배��ب�ADό�� ��Ŀ���t�؎%[�Ą���q7��T� �����k���S�*~`ߗ��uc��a0׫�@���[��<����^ǹmMJrH�Id��0�+ߠ���ml�R"BfNU����W�A�����t���UV@���̺*��7��~1����o��24�h��K㈶I5	(�!}'�eW��P����.'��F�<�n ӿ��{��6i;�,��r}^tǘ�UB@�O���L����q~[���ܑf�2�s��;�촲�yX�f|� ���E}�z'��V4�(��Cǎ-[�D�L�
x�{�\�=5.h�v�2*p����t��uT���o���2j}�s�5T���o˓�1Rk�P�C׎![���\��-1"k�P���&?�/���2	*y �'�eu��{�ܬ�>|�3ת!@��[�Ľl�R�t�X��Q\Æ.#�f	90�+�`�W��j/����"9&m%d�T�@���+�ഷȉiY3��0�����Ϸ��p��4����^/���F&M%
d�Խ`�Wہd���БcӖ"&s�����O�K�Ȫ)@����L�
x��|�^��JH��|�7��K���L�
x��x�]EL�
>x�s�Z!缵N˹h�Zs�����<�.c���9G�BzN\ˆ(�!Vg�_����h�QJC��k����z%��4�(�!Zg��\���=rnZS���v�>/����(�!M'�eX��`���!ng��R�s�Z3�����8�-NbK�H�	S�>zo���"!&g�D�̻������=O�KÈ�C���|�w��H�	D��:l��<�nS�B
Nx˝h�QQ�����L�
8���_�G�z\��8�-BbNVK���W�j���qz[�Ė,�"�r|��8��EBL�J+���w�`���}{�\��]3�j�s�=4�h��J��zu��%y$�$�d���;�,��}=nw��Rp�[���܏��4��qZ[�Ĝ��>s���� ����k�в#�f(�!`��QxÝnS��x��[�D���L�J9�9rmRt�X�EC��+�����Ie	�0�+�`����oӓ�6riQ4è�C��ۻ䌴���P��~+�����mc�VA2O�K�ȯ�C�;����t���UM ����g��|���5Jh��ic��2�x��_�G�x�]|�^-�}F^M�}X�Eg��Z ���r�yd��p����8�-\�F6M)
a8׭aBW�A[���ܔ��3��&0�+�ื�IjI�3�*1 ���{�ܦ&%<��4���X�����'�%x��t�X�\��>-/�c�V-"�_���r>Zo��ܢ&e=�p������{ߜ��a<׮!C��[�Ď,���p��t���]H�Im	y2]*F`��qpۛ䔴����R%d�T�������t�ؙeU������>��Ͼ+Ϡ����o�S�B3�j�����=RnBS�B�t����P����"3�j���:?���3�j3����;�l��"{�\��6>i/�#Ӧ"&|�4���AM�{�ܥf�<���z\��<�.c��8��X�E^LǊ-X�EvL�
%8��t�X�EX��j,��#�f9-0�k�P��>*o����-vbYE1�0���0��İ����uC��k�Ћ�ض%I$�$�$�$���đl��rrZZD�̬� ����������W�~�����aw��QU��������7��|����Ii	93�*`�W��ߟ���axםaVW�_����k�P���)�Ƿ�IrII4�(�!A'��[�Ĭ��~w��W�`�������1}+�`���Qc��7��J����w��{����9M-
bx�]aW�N��ؗp�[�Ī,���w�}5h��yS�~q��ԉ`��1`��{Ӝ�q=�t���W��K�J%��t��u[�ĕl���r6Zi�<���>o��̒*`�W�A|Ϟ+נ�GǍmZRD�L�J������yq�t��5U(��o���R=n~S���qg��T�����3�j6P��>;����.w��F0�k�о#Ϧ+� ���5nhӑb�rs����D��z���8�-S�BN}�x��QVC���̉j�3��=p�[���94�(�aJW��Yo���&:e,�� �g�i0�+㠶�=i.Q#��7��N1����_�G��t�X��_���-zb\�F!'�eL�� ���uo���b �g�w��K���H�	Ny�8�mQC�N
K�ȍiZQü��������8��_�G�M|�^(ǡmG�MRJBH�Ik��3�* ����yx�fvU �?���s�Z?��ܳ�
58��qr[�D�̐���/�c��>!/���F,�"*f`���w��|����J=�ys�t���L��h�x�z\�;�,�b�t���R�yn]�rvt��5D�̱j����R>Bo�Sۂ$�d���P���"w�YE��><�3ê. ���;�,�b�{���	=9.m#�fU2@�O���輱N����UZ@���� ����'��{�ܸ�E:L��2(�apכ�T���_��=t�X��J��)y!'�eI�0�+� ����n,Ӣ"f}p���T� �۟���x��RBq[������0��̀�����s�>t�J ���uq۵d�ԙ`���c��<�.��
}8�mg�UR@�O�K�L�
8��}r^ZG��\�F �?�o����<�n	�2*z`���!u'��Ut�د�C��8��@�O�K爵YH�	l�=2njS���r3�j����<�.c�V�;߬��n|Ӟ"�qE���8��P�C�N(ˡh��]S�Bv{��6�3�*;���zv\�r:Zl��,�bV}���W�q?��ԃ��7שaA��[˄���V�7ߩg�o����(�aI�1Y+� ���5a(סaG��QZC���� �?�/�#��8�-H�IvI	59(�!rg�UT�������:3��0�k������+�`��1jk�Г��6"i&Q%������봰���X�D�̾*�����~_�Gύk�P��Į,��w�Nu���}P�C�5[�āl���qb[�D����z:\��2-*b`�W�w��O�����I~I�7�)e!簵K�ȩiA�����|��?ׯ�C��9[��|�^��]K�H�	Vy?�o��21*k�Ї��v&Y%$��4����T�����'�uD�̥j����?�/�c�0�+Ӡ��}mRw�Y^E��Z>D�
$��t�X��T���/�#�6)<�.7��F?�o���"0�k����z*\��7�)raW��H��C�%;�섲�v�5e(��`���Qi�>���~(ߡgǕmP�C�N6K��9_��}n^S��^vG�U:@���+�`���!k���S��&.e#���3��)p����H�	Zy�<�n��:{�܂&e7��P�˾(��[Ǆ�\�FM7�iX�c��!;���B�v+� �'��c��8�-G�MFJM�yh�fs�����=L�J3��p����x�][�D��z�:	,�"=&ne���s��*$���x�U6@��;����{���V5?�