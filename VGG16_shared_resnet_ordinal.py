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
            # ri = (LOW_DIM*Ypred.shape[0])-(np.sum(se¾²{Ê\¨Æm?’oÒSâB6Ni‘8“­RB~N_‹‡ØeVTÁ ¯¿ÃÏî+ó ºÌıj>PïƒóŞ:'¬åB4Îh«‘@“Òât¶X‰Y<Å.,ã¢6i=.s£šı0¾kÏ«ÓÀ¢/Æcí2q*[ Ä‡ìrZqÛ¼¤›¼”›³ÔŠ ˜çÕu`Ø×åat×˜¡UG€Í_êGğÍ{ê\°Æí8²mJRHÂInI‰2*u ØçåutØØ¥eDÔÌ ªÀıoşSÿ‚?Şoç“õR8ÂmnRS‚BNw‹™X•PüÃş.?£¯Æí>2oªSÀÂ/îcó–:,ó¢:lı>rošSÔÂ ®gÃ•nÓ³â
6xéq6[©<Ÿ®Ã±n“¸’RzB\ÎF+ šgÔÕ` ×Çámw’YRELşJ?ˆïÙså4ôè¸±MKŠH˜ÉUi Ñ?ã¯öù>=/®cÃ–.#³¦
8üí~2_ªGÀÍoêSğÂ;îl³’
xò]zF\Í*} Şgç•uPØÃån4Ó¨¢FÚwäÙt¥„õ\¸Æm:RlÂR.BcV4Ÿ¨—ÁQoƒ“Ş'²eJTÈÀ©oÁï²3Êj(Ğác÷–9Q-¢~_½Î}kP—ƒÑ^#‡¦E6Lé
18ë­p‚[ŞD§Œ…ZÄö,¹"&zeÔö ¹'Í%jdĞÔ£à†7İ)fa°ñKûˆ¼™N°ø‹ıX¾EOŒËÚ(¤áD·Œ‰Zõ<¸îsºZÄú,¼â6{©6©7Á)o¡Ç²-JbHÖIa	¹1M+Š`˜×Õa`×—áQwƒ™^°ıKşH¿‰OÙå8´íH²IJIÉ9i-"s¦Züü¾>¯»ÃÌ®* şÿ½Î_ë‡ğ{Ö\¡½=NnK“ˆ’RuXşEŒßÚ'äåt´Øˆ¥YDÅ¬ú<şn?“¯Òâ~6_©Á=o®SÃ‚.c·–	Q9­>o¾SÏ‚+Ş`§—ÅQlÃ’.c²V
A8Ï­kÂP®CÃ.£´†9Vm²_ÊGèÍqj[Ä“ì’2jrPÚCäÎ4«¨€_ß‡çİufXÕ`ü×ş!§ŸÅWìÁr/šcÔÖ ¡'Ç¥mDÒL¢JHı	~y7Öia³±Jˆø™}U@÷ù[ı¾|×´¡H‡‰]YE=îz3œê0ñ+û ¼‡Îk¶P‰Ù>%/¤ãÄ¶,‰"&u%äõt¸ØeZTÄÀ¬¯Âî~3ŸªÀñoû“ü’>o²SÊB(Îak—‘SÓ‚"¿Òâ{ö\¹=:nlÓ’"frU@ôÏø«ı@¾OÏ‹ëØ°¥KÄÈ¬©B»ŸÌ—êpó›ú¼ğ;Û¬¤‚|—W³Jˆ÷Ùyeöp¹Í4ªh€Ñ_ã‡öy6])a=®qC››°”‹Ğ˜£ÕF Í'êepÔÛà¤·Ä‰l™2pê[ğÄ»ìŒ²
tøØ½eNTË€¨ŸÁWïsßš'Ôå`´×È¡iG‘SºBÎz+œà–7Ñ)c¡±=K®Hƒ‰^µ=HîIs‰4õ(¸áMwŠYXÅlüÒ>"o¦SÅ,şb?–oÑã²6
i8Ñ-c¢VA=®{Ãœ®±>¯¸ƒÍ^*G ÍGêMpÊ[èÄ±l‹’’uRXÂEnLÓŠ"æuuØõexÔİ`¦WÅlÿ’?ÒoâSöB9m;’l’RBrNZK„Èœ©V?¿¯ÏÃëî0³«Ê ¨ÿÁïŸó×ú!|ç5W¨ÁAo“ÛÒ$¢d†T –Ñã·ö	y9-6biQ1«¾ ¿ÛÏä«ô€¸ŸÍWêApÏ›ëÔ° ‹ÇØ­eBTÎ@«À›ïÔ³àŠ7ØéeqÛ°¤‹Ä˜¬•BÎsëš0”ëĞ°£ËÆ(­!BgU[€ÄŸì—òzsœÚ$ñ$»¤Œ„š”ö¹3Í**`à×÷áywVuÿµÈßégñ{°Ü‹æµ5HèÉqi‘4“¨’R‚_ŞGçuZXÄÅl¬Ò"~f_•ĞıcşV?/ß£çÆ5m(ÒabW–AQƒ»Ş§ºLüÊ>(ï¡sÇš-Tâ@¶OÉé8±-K¢H†I]	y=.vc™10ë«ğ€»ßÌ§êpüÛş$¿¤Ä›ì”²ŠsØÚ%däÔ´ ˆ‡Ù]eTı ¾ÏŸë×ğ¡{Çœ­VA>O¯‹ÃØ®%C¤Î«¼€Û·ä‰t™•5PèÃñn;“¬’~r_šGÔÍ`ªWÀÁoï“óÒ:"læR5hşQƒŸŞç±uK˜È•iPÑã¾6©;Á,¯¢Æ~-¢wÆYm|ò^:G¬ÍB*N`Ë—è‘qS›‚p—›ÑT£€†İ7æiuóµzÜùf=.pã›ö¹0+Ú`¤×Ä¡l‡’RvBYE;Œìš2êp°ÛËä¨´HŸ‰WÙe?”ïĞ³ãÊ6(é!q'›¥T„Àœ¯Öá>7¯©CÁ/»£Ì†* ögù}0ŞkçµSÈÂ)na—²JsˆÚdõ¸ğ{Ú\¤Æ­<‚nS·‚	^y=VnA²Êt¨Øe_”ÇĞ­cÂV.A#?ÒoâSöB9m;’l’RBrNZK„Èœ©V?¿¯ÏÃëî0³«Ê ¨ÿÁïŸó×ú!|ç5W¨ÁAo“ÛÒ$¢d†T –Ñã·ö	y9-6biQ1«¾ ¿ÛÏä«ô€¸ŸÍWêApÏ›ëÔ° ‹ÇØ­eBTÎ@«À›ïÔ³àŠ7ØéeqÛ°¤‹Ä˜¬•BÎsëš0”ëĞ°£ËÆ(­!BgU[€ÄŸì—òzsœÚ$ñ$»¤Œ„š”ö¹3Í**`à×÷áywVuÿµÈßégñ{°Ü‹æµ5HèÉqi‘4“¨’R‚_ŞGçuZXÄÅl¬Ò"~f_•ĞıcşV?/ß£çÆ5m(ÒabW–AQƒ»Ş§ºLüÊ>(ï¡sÇš-Tâ@¶OÉé8±-K¢H†I]	y=.vc™10ë«ğ€»ßÌ§êpüÛş$¿¤Ä›ì”²ŠsØÚ%däÔ´ ˆ‡Ù]eTı ¾ÏŸë×ğ¡{Çœ­VA>O¯‹ÃØ®%C¤Î«¼€Û·ä‰t™•5PèÃñn;“¬’~r_šGÔÍ`ªWÀÁoï“óÒ:"læR5hşQƒŸŞç±uK˜È•iPÑã¾6©;Á,¯¢Æ~-¢wÆYm|ò^:G¬ÍB*N`Ë—è‘qS›‚p—›ÑT£€†İ7æiuóµzÜùf=.pã›ö¹0+Ú`¤×Ä¡l‡’RvBYE;Œìš2êp°ÛËä¨´HŸ‰WÙe?”ïĞ³ãÊ6(é!q'›¥T„Àœ¯Öá>7¯©CÁ/»£Ì†* ögù}0ŞkçµSÈÂ)na—²JsˆÚdõ¸ğ{Ú\¤Æ­<‚nS·‚	^y=VnA²Êt¨Øe_”ÇĞ­cÂV.A#¦Å4¬è‚1^k‡SÖB!g»•LÊèò1zkœĞ–#Ñ&#¥&å<´î³¹Júy|İ&w¥Dõ¸ú|ú^<Ç®-C¢NK½y[–|‘·²	Jyİ9fmpò[úD¼Ì* ô‡ø}V^A½[ÎD«Œ€šÔ÷à¹wÍjuØóåz4Üè¦1E+Œàš7Ôé`±Ë±h‹‘X“…RÂv.Y#…&å64é(±!K§ˆ…Y\Å,ı">fo•Ğò#úf<Õ. ã§öy<İ.&c¥ñ<»®ƒº÷º9Lí
2xê]pÆ[í²|Š^ÇµmHÒIbII1	+¹ 'ÚedÔÔ  ‡ÇİmfRU@şOÿ‹ÿØ¿åOôËø¨½ANO‹‹Ø˜¥UDÀÌ¯êğş;ÿ¬¿Â>o²SÊB(Îak—‘SÓ‚"fw•Põøş=®_Ã‡îs¶Z	ù<½.c»–‘:¬ò:~lß’'ÒebTÖ@¡Ç»íL²J
HøÉ}iQ7ƒ©^¿½OÎKëˆ°™KÕ ùGı~z_œÇÖ-a"W¦AEŒûÚ<¤î³¼Šûµ|ˆŞgµHğÉ{é±6©8-_¢GÆMm
RxÂ]nFS~tß˜§ÕE`Ì×ê!pç›õT¸ÀoÚSäÂ4®hƒ‘^‡²JvHÙ	e9í0²kÊP¨ÃÁn/“£Ò"}&^e”ıP¾CÏ+Û ¤‡Äl–Rs¾Z„ûÜ¼¦;¼ì2ªt€ØŸåWôÁx¯CÖN!§¸…M\ÊF(Í!jgÕSàÂ7îis‘´òºyLİ
&xåtöX¹M<Ên(Ó¡b–}QC·	[¹<šnÓ°¢Æx­BvNY…8œíV2A*O ËÇè­qB[D›Œ”š”óĞº#Ìæ*5 èçñu{˜Ü•fÕ3àê7ğé{ñ»¶‰:,õ"8æmuXòEzLÜÊ&(å!tç˜µUHÀÉoéñ2;ªl€ÒâwöYy<ön9­2j~PßƒçŞ5g¨ÕA`Ï—ëÑp£›Æ­0‚kŞP§ƒÅ^,Ç¢-FbMJqÛ¹dšp”ÛĞ¤£Ä†,"fq°ô‹ø˜½UN@Ëè›ñT»€ŒŸÚäñt»˜Œ•ZÄóìº2êz0Üëæ0µ+Èà©wÁoµÈò)za×¶!I'‰%Y$Å$¬ä‚4h—‘QSƒ‚w·™IU	 ù?ı/şcÿ–?Ñ/ã£ö9=-.bc–V3¿ªÀûïü³ş
?¸ïÍsêZ0Äëì°²Êx¨İAfO•Ğø£ıF>M/ŠcØÖ%a$×¤¡D‡ŒZDñ»ºŒú<ôî8³­JHşI‰Ù7å)tá·µIHÉ	i9-3¢jPış~?Ÿ¯×Ãán7“©R¾_Ï‡ëİp¦[Å¬ü‚>o·“ÉR)a>W¯Cß'Û¥d„Ôœ –Ñ=c®V>¯·ÃÉn)¡2ª}@ŞOç‹õX¸ÅMlÊR(ÂanW“R‚wŞYg…\ğÆ;í,²b
VxÁo¶SÉ)>a/—£ÑF#&e4Ôè ±GËhšQTÃ€®Ã·î	s¹4úh¼ÑN#‹¦…5\èÆ1m+’`’WÒAbO–KÑ£¹F:zlÜÒ&"e&Tå ´ÿÈ¿éOñû¸¼NK´Èˆ©YA¼ûÎ<«® ƒ¿Şç»õL¸ÊhúQ|Ã<’nS²B
NxËh–QQƒ¾·»ÉL©
8ÿ­Â_îGóz\ôÆ8­-BbNVKŸ¹WÍjßÓçâ5vhÙe3”ê°óËú(¼áN7‹©X_¼ÇÎ-k¢P†Cİ&{¥„ö¹6):a,×¢!FgZpÄÛì¤²Š|˜Şg°ÕKàÈ·éIq	¹4(šaT×€¡_Ç‡í]rFZMÊ|¨Şg¿•OĞËãè¶1I+‰ ™'Õ%`ä×ô¡x‡]VFAº{ÌÜª& å?ôïø³ıJ>Hï‰sÙ%4äè´±H‹‰X™U<Àî/ó£ú<ı.>c¯–Ñ>#¯¦Å>,ï¢3Æj-âsöZ9í<²n
S¸ÂnzSœÂ.q#›¦…0œëÖ0¡+Ç ­GÂMnJSˆÂnu˜òzpÜÛæ$µ$ˆä™t•õSøÂ=nnS“‚rwšYTÅ ¬ÿÂ?îoó“ú<òn:S¬Â.~cŸ–Ñ1c«– ‘?Ó¯âö~9­7ÂinQƒ²
w¸ÙMe
TøÀ½oÎSë‚0k×¡SÇ‚-^bG–MQ
C¸ÎkºPŒÃÚ.$ã¤¶‰<™.#°æõ8¸íMrJZHÄÉl©2ª_ÀÇïísòZ:DìÌ²*
`ø×ıa~WŸWßgß•gĞÕcàÖ7á)w¡GµHúI|É)7¡)G¡GºMLÊJ(Èáiw‘SµşyÖwáwµHõ	xù}6^i‘=S®B~Ÿ´—È‘iS‘¾rš{ÔÜ ¦Å=lîR3‚jP÷ƒù^=®}CN‹±X‹…XœÅV,Á"/¦cÅ,ñ";¦l…òv:Y,Å",æb5hñ{³œŠñ5{¨Üf•7Ğécñ;±,‹¢†u]ÆumÒubXÖEa×º!LçŠ5XèÅqlÛ’$’d’T’@’OÒKâH¶II		99--"bfVU ÿ¿ÿÏÿëÿğ¿ûÏü«ş ¿¿ÏÏëëğ°»ËÌ¨ª@ÿÿÛÿä¿ôø›ıT¾@ÛÛä¤´„ˆœ™V0ÿ«ÿÀ¿ïÏóëú0¼ëÎ0««À€¯ßÃçî5s¨Údÿ”¿ĞãÛö$¹$$šd”Ô “ÇÒ-bbVVA¿»ÏÌ«ê °ÿËÿè¿ñOû‹ü˜¾O°ËËè¨±AKˆ›ÙT¥ „ÿÜ¿æõ;øì½rZ{„Üœ¦1<ë®0ƒ«Ş §¿ÅOìËò(ºaL×Š!Xç…u\ØÆ%m$Òd¢T†@Ö{á·¶	I9	-9"m&ReTş@¿ÏÛëä°´‹È˜©UA Ï¿ëÏğ«ûÀ¼=RnBSBt›˜”•PÃÓî"3¦jüóş:?¬ïÂ3îj3êğò;úl¼Ò"{¦\…ı6>i/‘#Ó¦"&|å4÷¨¹AMŠ{ØÜ¥fÕ< îó½z\û†<.c±±8‹­X‚E^LÇŠ-XâEvLÙ
%8äít²XŠEXÌÅj,Ğâ#öf9-0âköP¹Í>*o ÓÇâ-vbYE1ëº0ŒëÚ0¤ëÄ°¬‹Â®uC˜Îk°Ğ‹ãØ¶%I$É$©$$Ÿ¤—Ä‘l“’rrZZDÄÌ¬ª şÿŸÿ×ÿá÷ŸùWı~ŸŸ××áaw—™QU€şÿ·ÿÉéñ7û©|··ÉIi	93­*`şWÿßŸç×õax×aVW_¿‡ÏİkæPµÈş)¡Ç·íIrII4É(©!A'¥[ÄÄ¬¬‚~wŸ™WÕ`ÿ—ÿÑãŸöù1}+`——ÑQcƒ–7³©Jÿ¹ÍêwğÙ{å´ö¹9M-
bxÖ]aW½N‹ŸØ—åQtÃ˜®C°Îë¸°KÚH¤ÉD©:¬÷Â9nm’rZrDÚL¤Ê¨ü~Ÿ·×Éai‘1S«‚ ×ŸáW÷y_Ö}aW·I_‰Ù=e.Tã€¶É7é)q!§´…HœÉV)!?§¯ÅCìÎ2+ª`€×ßág÷•yPİæ~5¨÷ÁyoÖr!g´ÕH ÉGéq:[¬Ä‚,b–qQƒ´—¹QMŠ~ßµgÈÕi`Ñã±v™8•-PâCöN9­8‚m^RG‚M^JGˆÍYjEÌóê:0ìëò0ºkÌĞª#Àæ/õ#øæ=u.Xã…vÙ6%)$á$·¤‰D™•:ìóò::lìÒ2"jfPÕàş7ÿ©Áï·óÉz)á67©)A!§»ÅL¬Ê(şa—ŸÑWãv™7Õ)`á÷±yK–yQ¶~	¹7Í)ja×³áJ7ˆéYq¼ô8›­T‚@O×‹áX·…I\É)=!.g£•FÍ3êj0Ğëãğ¶;É,©"&¥Ä÷ì¹rztÜØ¦%E$Ìäª4€èŸñWû|Ÿ×±aK—ˆ‘YS…şv?™/Õ#àæ7õ)xáw¶YI	<ù.=#®f•>ï³óÊ:(ìár7šiTÑ £¿Æí;òlºRÂz.\ã†6)6a)¡1G«@šOÔËà¨·ÁIo‰Ù2%*dàÔ·à‰wÙe5èğ±{Ëœ¨–Q?ƒ¯Şç¾5O¨ËÁh¯‘CÓ"¦t…œõV8Á-9RmR~B_GÛdšT”À¯ÓÃâ.6c©1?«¯ÀƒïŞ3çª5@èÏñkû¼“Î+²`ŠWØÁeo”ÓĞ¢#Æf-"pæ[õ¸ü~_´ÇÈ­iBQC»›ºŒğš;Ôì ²Ê}hŞQgƒ•^Ç³íJ2HêIpÉé4±(‹¡X‡…]\ÆF-"zf\Õ ı'şe”ßĞ§ãÅv,Ù"%&då´ğˆ»ÙL¥
øü½~_»‡ÌjPñû¾<®Ã´®ƒ¹^º}LŞJ'ˆåYtÅ¬õB8Îmk’P’CÒN"K¦H…	\ù==.nc“–2sªZ Äÿì¿òú{üÜ¾&¥;Äì¬²
~xßgÖUa ×¿áO÷‹ùX½N|Ë(—¡QGƒ^G´ÍHªI@Éé;ñ,»¢†zöv9-5"hæQu˜ş°ßËçèµqHÛ‰d™•0ëÓğ¢;Æl­r~Z_„ÇÜ­fU>@ïóÛú$¼ä4›¨”PŸƒ×Ş!g§•EPÌÃê.0ã«ö ¹?Í/êcğÖ;á,·¢	Fy:vlÙ%2dêT°À‹ïØ³åJ4Èè©qA´›È”©Pß¾'Ï¥kÄĞ¬£Â.}#f•1Pëƒğ;×¬¡B}[D—Œ‘Z„òºvÙ:%,äâ4¶h‰Y3…*àö7ù)}!g·•IPÉé>1/«£À†/İ#æf5(ğá{÷œ¹V:¬ßÂ'îes”Ú¤óÄº,Œâ6té±5K¨Èi_‘Ó½bV{Ÿ¶É1i+‘ “§Òb|Ö^!§½ENLËŠ(˜áUw€Ù_åôıx¾]O†Kİ¦yEöz9í62i*Q Ã§îs¼Ú$û¤¼„›¶‰0™+Õ  çÇõmxÒ]bFVM
¸ßÍgêUpÀÛïä³ôŠ8˜íUr@ÚOäËô¨¸M_ŠGØÍejTĞÀ£ïÆ3í*2`êWğÁ{ïœ³Ö
!8ç­uBXÎEkŒĞš#Ôæ µ'ÈåitÑ£µFÍ9jmÒsâZ6Dé±:¬ø‚=^nG“RBtÎX«…@œÏÖ+á ·§ÉEiÑ:#¬æ5>hï‘sÓš"æpµÈô©x_¶GÉi:Q,Ã¢.c½q;›¬”‚s×š!Tç€µ_ÈÇémq[²DŠL˜ÊhğÑ{ãœ¶	19+­ ‚gŞUg€Õ_àÇ÷íyr]FtÍªu@ØÏåkôĞ¸£ÍF*M ÊgèÕq`Û—ä‘t“˜’RpÂ[îD³ŒŠôõx¸İMfJUÀùoışr?šoÔÓà¢7Æi8’mRRBBNNK‹ˆ˜™UU Àÿïÿóÿú?üïş3ÿª?Àïïóóú:<ìî23ªj Ğÿãÿö?ù/ı#şf?•/Ğããö69)-!"g¦UE Ìÿê?ğïûóüº>ïº3Ìê*0àë÷ğ¹{Íªv Ù?å/ôãø¶=I.I#‰&%5$èä±t‹˜˜•UPÀÃïî3óª: ìÿò?úoüÓş"?¦oÅìò2:jlĞÒ#âf6U) á?÷¯ùCı>{¯œƒÖ!7§©EAÏº+Ìàª7Àéoñû²<ŠnÓµbÖya¶qI‰4™(•