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
        #                                      validation_data=gen8„í\²F
M8ÊmhÒQbC–N³¸ŠXúE|ÌŞ*' åGôÍxª]@ÆOíòxº]LÆJ-âyv]u=îus˜ÚdğÔ»àŒ·Ú	dù½0kÛ¤“Ä’,’bVrAO´ËÈ¨©AA»ÛÌ¤ª€üŸşÿ±ËŸè—ñQ{ƒœ±1K«ˆ€™_ÕàıwşY…Ü÷æ9u-âuvXÙe<Ôî ³§ÊhüÑ~#Ÿ¦Å1lë’0’kÒP¢CÆN-¢x†]]F}zwœÙV%$ÿ¤¿Äì›òºpŒÛÚ$¤ä„´œˆ–Q5¨ş¿ŸÏ×ëáp·›ÉT© ?ß¯çÃõn8Ó­bV~A·ÛÉd©0Ÿ«×À¡oÇ“íR2BjNPËƒè1W«@Ÿ×Ûád·”‰P™Õ> ï§óÅz,Üâ&6e)á0·«É@©Á;ï¬³Â
.xãvY1+¼à7Û©dŸ°—ËÑh£‘F2jtĞØ£åF4Í(ªa@×á[÷„¹\}4Şh§‘ESŒÂ.tã˜¶I0É+é ±'Ë¥h„Ñ\£†=6ni‘2ªr Úäßô§ø…}\ŞF'%ZdÄÔ¬ ‚Ş}gUW€Á_ï‡óİz&\å4ı(¾aO—‹ÑX£…FÍ6*i Ñ'ã¥vÙ<¥.ã¼¶	;¹,"ftÕ õGøÍ}j^PÇƒí^2GªM@ÊOèËñh»‘L“ŠòuzXÜÅf,Õ" ægõxğİ{æ\µı9~m’wÒYbELñ
;¸ìrZtÄØ¬¥BÎ|« —¿ÑOã‹ö¹5M(Êah×‘aS—‚^s‡šTö@¹Í;êl°Òâx¶]II=	.y#&e1ë°°‹ËØ¨¥ADÏŒ«Ú ¤ÿÄ¿ìòút¼Ø%[¤Ä„¬œ‚q7›©T Ÿ¿×Ïák÷¹SÍ*~`ß—çÑuc˜Öa0×«á@·É[é±<‹®ƒµ^Ç¹mMJrHÚIdÉ©0+ß §ÇÅmlÒR"BfNU€øŸıWşAŸÛ×ä¡t‡˜UV@Áï»óÌº*àú7üé~1«·À‰oÙå24êh°ÑKãˆ¶I5	(ù!}'eW”ÁP¯ƒÃŞ.'£¥FÍ<ªn Ó¿âö{ù½6i;‘,“¢r}^tÇ˜­UB@ÎOë‹ğ˜»ÕL Êèıq~[Ÿ„—Ü‘f•2êsğÚ;äì´²ŠyXİf|Õ ÷§ùE}Şz'œåV4Á(¯¡CÇ-[¢D†L
xñ{¶\‰=5.hã‘v™2*pà9Dí²z
\øÆ=m.Rc‚VA7©[Á¯¼ƒÎ+· ‰GÙe:TìÀ²/ÊcèÖ1a+— ‘GÓbVtÁ¯µCÈÎ)k¡‡³İJ&Hå	tù½5NhË‘h“‘R‚rZw„Ù\¥ı<¾n“»Ò¢z\ı>}/c×–!Q'ƒ¥^Ç¼­NK¾H‰[Ù¥<„î³¶
	8ù-}"^fG•PúCüÎ>+¯ ƒÇŞ-g¢UF@Íê{ğÜ»æµ:ìùr=ntÓ˜¢FpÍêt°Ø‹åX´ÅH¬ÉB)a;—¬‘Bršt”Ø¥SÄÂ,®b–~³·Ê	hù}3jñSû‚<n“±R‚x]W†A]†{İ¦v<õ.8ã­vY>E/ŒãÚ6$é$±$‹¤˜„•\Æí22jjPĞÃãî63©* ÿ§ÿÅìßò'úe|ÔŞ §§ÅElÌÒ*"`æWõxÿÖ_á÷½yN]†xVvAµ;Èì©r´ßÈ§éEqÛº$Œäš4”è±SË‚(aW—Q_ƒ‡Şg¶UI É?é/ñ#û¦<….ã¶6	)9!-'¢eFTÍ ªÀßïçóõz8Üíf2U*@àÏ÷ëùp½Ît«˜€•_ĞÇãív2Y*E Ìçê5pèÛñd»”ŒšÔò ºgÌÕj Ğçãõv8Ù-e"Tæ@µÈûé|±·¸‰MY
E8Ìíj2PêCğÎ;ë¬°‚Şx§EVLÁ
/¸ãÍv*Y Å'ìår4Úh¤ÑD£Œ†4öh¹M3ŠjĞõcøÖ=a.W£F7ÚidÑ£°†İ8¦mELòJ:HìÉr)a4×¨¡AG[ÚD¤Ì„ª€öù7ı)~a—·ÑIc‰15+¨àwß™gÕ`ğ×ûá|·	W¹M?ŠoØÓåb4Öh¡G³JHôÉx©A6O©Á8¯­CÂN.K£ˆ†]5hı~sŸšÔñ`»—Ì‘jòúr<Ún$Ó¤¢†|w±KµˆùY}|÷9W­B_Û‡ät–X‘S¼Â.{£œ†16k©3ßª'ÀåoôÓø¢=FnMŠrÚudØÔ¥`„×Ü¡f•=PîCó:¬ô‚8mW’ARO‚KŞH§‰EYÅ:,ìâ26jiÑ3ãª6 é?ñ/û£ü†>/¶cÉ)1!+§ …GÜÍf*U ÀçïõsøÚ=dîT³€ŠØ÷åytİ¦uEÌõj8ĞícòV:A,Ï¢+Æ`­Âqn[“„’’vY2E*LàÊ7èéqq=DîL³Š
øõ}xŞ]g†U] ÆíòwúY|Å,÷¢9Fmzr\ÚF$Í$ªd€ÔŸà—÷Ñycq1«´€ˆŸÙWåtÿ˜¿ÕOàË÷è¹qMŠt˜Ø•ePÔÃà®7Ã©n¿²Ê{èÜ±f•8íSòB:NlË’(’aRW‚A^O‡‹İX¦EEÌú*<àî7ó©zÿ¶?É/é#ñ&;¥,„â¶v	95-(âavW™U?€ïßóçú5|èŞ1g«•@ÏÓëâ0¶kÉ©3Á*/ ãÇö-y"]&Feúp¼ÛÎ$«¤€„ŸÜ—æu3˜êpğÛûä¼´›¹T šÔßà§÷Åylİ&reTôÀ¸¯ÍCêN0Ë«è€±_Ë‡èqV[Ÿ¼—Îk³ŠØò%zdÜÔ¦ …'Üåf4Õ( áG÷yZ]Æ|­w¾YO…Üø¦=E.LãŠ6é5q(Û¡d‡”P–CÑ#»¦…:ìö29*m ÒgâUv@Ùå;ôì¸²JzHÜÉf)!0ç«õ@¸ÏÍkêP°ÃËî(³¡JˆıY~EŒ÷Ú9dí²pŠ[ØÄ¥l„Ò¢vY=.|ã6©1A+ ›ÇÔ­`‚WŞAg•[ĞÄ£ì†2*v`Ùå1të˜°•KĞÈ£éF1+º`Œ×Ú!dç”µPˆÃÙn%¤òº|ŒŞ'´åH´ÉH©	A9­;Âl®R‚~_·‡É]iQ=®~Ÿ¾Ï±kË¨“ÁR/‚cŞV'%_¤ÇÄ­l‚RBwY[…œü–>/³£Ê(ı!~gŸ•WĞÁcï–3Ñ*# æõ=xî]s†Zö|¹7ºiLÑ
#¸æu:XìÅr,Úb$Öd¡‡°KÖH¡	G¹M:JlÈÒ)baW±K¿ˆÙ[å´üˆ¾OµÈø©}AO·‹ÉX©A<Ï®+Ã ®Ã½nS»‚zœñV;,Ÿ¢Æqm’t’X’ERLÂJ.Hã‰v55(èáqw›™T• ÿÓÿâ?öoùı2>joÓÓâ"6fi0ó«ú ¼ÿÎ?ë¯ğƒûŞ<§®C¼Î+» Œ‡ÚdöT¹ ?ÚoäÓô¢8†m]FrMJtÈØ©eAÏ°«ËÀ¨¯ÁCï3Ûª$€äŸô—ø‘}SBq[›„”œ–Ñ2#ªf Õ?àï÷óùz=îv3™* ğçûõ|¸ŞgºULÀÊ/èãñv;™,•"æsõ8ôíx²]JFHÍ	jyİ3æj5èóñz;œì–2*s Úäıt¾X…[ÜÄ¦,…"æv5<„î³¶
	8ù-}"^fG•PúCüÎ>+¯ ƒÇŞ-g¢UF@Íê{ğÜ»æµ:ìùr=ntÓ˜¢FpÍêt°Ø‹åX´ÅH¬ÉB)a;—¬‘Bršt”Ø¥SÄÂ,®b–~³·Ê	hù}3jñSû‚<n“±R‚x]W†A]†{İ¦v<õ.8ã­vY>E/ŒãÚ6$é$±$‹¤˜„•\Æí22jjPĞÃãî63©