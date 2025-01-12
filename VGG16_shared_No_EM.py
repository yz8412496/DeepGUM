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
from keras.applications.vgg16 import VGG16
from VGG16_rn import extract_XY_generator, extract_XY_generator_multiple
from data_generator_shared_No_EM import load_data_generator,load_data_generator_simple
from data_generator_shared_No_EM import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
from scipy.special import logsumexp
from log_gauss_densities import gausspdf,loggausspdf
from test import run_eval

start_time = time.time()
encode = "ISO-8859-1"
Age = "test" # "C"
# DISPLAY_TEST=True
DISPLAY_TEST=False
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
ITER = 1
# ITER = 6
WIDTH = 112
BATCH_SIZE = 64
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
        model_vgg16 = VGG16(input_shape=(112,112,3), include_top=False, weights='imagenet')
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

        # Alternative training
        # model.layers[20].trainable = False
        # model.layers[22].trainable = False
        # model.layers[24].trainable = False
        # model.layers[26].trainable = False

        # model.layers[21].trainable = False
        # model.layers[23].trainable = False
        # model.layers[25].trainable = False

        # rn = Input(shape=(LOW_DIM,), name='input_2')
        # weightedReg = merge([rn,reg], mode='mul', name='merge')
        # modelRn = Model(input=[model_vgg16.input,rn], output=weightedRn)
        # modelRn = Model(input=[model_vgg16.input,rn], output=[cla,weightedReg])

        self.network = model

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
                  decay=1e-06)
        adam = Adam(lr=learning_rate)
        rms = RMSprop(lr=learning_rate)
        self.network.summary()
        # self.networkRn.summary()
        
        # load previous trained weights
        # self.network.load_weights(self.fileWbest)

        # self.networkRn.compile(optimizer=rms, loss={'classification': 'categorical_crossentropy', 'merge': 'mse'},
        #     metrics={'classification': 'accuracy', 'merge': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/1000])
        self.network.compile(optimizer=sgd, loss={'classification': 'categorical_crossentropy', 'regression': 'mse'},
            metrics={'classification': 'accuracy', 'regression': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/1000])

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


    # def custom_loss(self):
    #     def loss(y_true, y_pred):
    #         # return -K.mean(K.sum(self.networkRn.input[1]*y_true*K.log(y_pred), axis=-1))

    #         p = self.networkRn.input[1]
    #         y_pred = y_pred + 1e-15
    #         return -K.sum(p*y_true*K.log(y_pred), axis=-1)
    #         # return -K.sum(p*y_true*K.log(y_pred), axis=-1)
    #         # L_reg = K.square(y_pred_0*p - y_true_0*p)
    #         # L_cla = -K.sum(p*y_true_1*K.log(y_pred_1), axis=-1)

    #         # return alpha*L_reg + (1-alpha)*L_cla
    #     return loss


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

        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06)
        adam = Adam(lr=learning_rate)
        rms = RMSprop(lr=learning_rate)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        # self.networkRn.compile(optimizer=rms, loss={'classification': self.custom_loss(), 'merge': 'mse'},
        #     metrics={'classification': 'accuracy', 'merge': 'mean_squared_error'}, loss_weights=[1-alpha, alpha/1000])
   £ú<ı.>c¯–Ñ>#¯¦Å>,ï¢3Æj-âsöZ9í<²n
S¸ÂnzSœÂ.q#›¦…0œëÖ0¡+Ç ­GÂMnJSˆÂnu˜òzpÜÛæ$µ$ˆä™t•õSøÂ=nnS“‚rwšYTÅ ¬ÿÂ?îoó“ú<òn:S¬Â.~cŸ–Ñ1c«– ‘?Ó¯âö~9­7ÂinQƒ²
w¸ÙMe
TøÀ½oÎSë‚0k×¡SÇ‚-^bG–MQ
C¸ÎkºPŒÃÚ.$ã¤¶‰<™.#°æõ8¸íMrJZHÄÉl©2ª_ÀÇïísòZ:DìÌ²*
`ø×ıa~WŸWßgß•gĞÕcàÖ7á)w¡GµHúI|É)7¡)G¡GºMLÊJ(Èáiw‘SµşyÖwáwµHõ	xù}6^i‘=S®B~Ÿ´—È‘iS‘¾rš{ÔÜ ¦Å=lîR3‚jP÷ƒù^=®}CN‹±X‹…XœÅV,Á"/¦cÅ,ñ";¦l…òv:Y,Å",æb5hñ{³œŠñ5{¨Üf•7Ğécñ;±,‹¢†u]ÆumÒubXÖEa×º!LçŠ5XèÅqlÛ’$’d’T’@’OÒKâH¶II		99--"bfVU ÿ¿ÿÏÿëÿğ¿ûÏü«ş ¿¿ÏÏëëğ°»ËÌ¨ª@ÿÿÛÿä¿ôø›ıT¾@ÛÛä¤´„ˆœ™V0ÿ«ÿÀ¿ïÏóëú0¼ëÎ0««À€¯ßÃçî5s¨Údÿ”¿ĞãÛö$¹$$šd”Ô “ÇÒ-bbVVA¿»ÏÌ«ê °ÿËÿè¿ñOû‹ü˜¾O°ËËè¨±AKˆ›ÙT¥ „ÿÜ¿æõ;øì½rZ{„Üœ¦1<ë®0ƒ«Ş §¿ÅOìËò(ºaL×Š!Xç…u\ØÆ%m$Òd¢T†@Ö{á·¶	I9	-9"m&ReTş@¿ÏÛëä°´‹È˜©UA Ï¿ëÏğ«ûÀ¼¯Îë¾0«ÛÀ¤¯Äƒì2ªq@Ûä›ô”¸SÚB$Îd«”€ŸÓ×â!vg™U0Àëïğ³ûÊ<¨îs¿šÔûà¼·Î	k¹3Új$Ğä£ô†8-VbAO±Ë¸¨AZO„ËÜ¨¦E?ŒïÚ3äê4°è‹ñX»…LœÊ(ñ!{§œ…VÁ6/©#Á&/¥#Äæ,µ"æyuöuyİ5fhÕ`ó—ú|ó:¬ñB;l›’’p’[ÒD¢L†Jöyy6vi53¨êpÿ›ÿÔ¿à÷Ûùd½p››Ô” ‡Óİb&Ve¢:lı>rošSÔÂ ®gÃ•nÓ³â
6xéq6[©<Ÿ®Ã±n“¸’RzB\ÎF+ šgÔÕ` ×Çámw’YRELşJ?ˆïÙså4ôè¸±MKŠH˜ÉUi Ñ?ã¯öù>=/®cÃ–.#³¦
8üí~2_ªGÀÍoêSğÂ;îl³’
xò]zF\Í*} Şgç•uPØÃån4Ó¨¢FÚwäÙt¥„õ\¸Æm:RlÂR.BcV4Ÿ¨—ÁQoƒ“Ş'²eJTÈÀ©oÁï²3Êj(Ğác÷–9Q-¢~_½Î}kP—ƒÑ^#‡¦E6Lé
18ë­p‚[ŞD§Œ…ZÄö,¹"&zeÔö ¹'Í%jdĞÔ£à†7İ)fa°ñKûˆ¼™N°ø‹ıX¾EOŒËÚ(¤áD·Œ‰Zõ<¸îsºZÄú,¼â6{©6©7Á)o¡Ç²-JbHÖIa	¹1M+Š`˜×Õa`×—áQwƒ™^°ıKşH¿‰OÙå8´íH²IJIÉ9i-"s¦Züü¾>¯»ÃÌ®* şÿ½Î_ë‡ğ{Ö\¡½=NnK“ˆ’RuXşEŒßÚ'äåt´Øˆ¥YDÅ¬ú<şn?“¯Òâ~6_©Á=o®SÃ‚.c·–	Q9­>o¾SÏ‚+Ş`§—ÅQlÃ’.c²V
A8Ï­kÂP®CÃ.£´†9Vm²_ÊGèÍqj[Ä“ì’2jrPÚCäÎ4«¨€_ß‡çİufXÕ`ü×ş!§ŸÅWìÁr/šcÔÖ ¡'Ç¥mDÒL¢JHı	~y7Öia³±Jˆø™}U@÷ù[ı¾|×´¡H‡‰]YE=îz3œê0ñ+û ¼‡Îk¶P‰Ù>%/¤ãÄ¶,‰"&u%äõt¸ØeZTÄÀ¬¯Âî~3ŸªÀñoû“ü’>o²SÊB(Îak—‘SÓ‚"fw•Põøş=®_Ã‡îs¶Z	ù<½.c»–‘:¬ò:~lß’'ÒebTÖ@¡Ç»íL²J
HøÉ}iQ7ƒ©^¿½OÎKëˆ°™KÕ ùGı~z_œÇÖ-a"W¦AEŒûÚ<¤î³¼Šûµ|ˆŞgµHğÉ{é±6©8-_¢GÆMm
RxÂ]nFS~tß˜§ÕE`Ì×ê!pç›õT¸ÀoÚSäÂ4®hƒ‘^‡²JvHÙ	e9í0²kÊP¨ÃÁn/“£Ò"}&^e”ıP¾CÏ+Û ¤‡Äl–Rs¾Z„ûÜ¼¦;¼ì2ªt€ØŸåWô ºÌıj>PïƒóŞ:'¬åB4Îh«‘@“Òât¶X‰Y<Å.,ã¢6i=.s£šı0¾kÏ«ÓÀ¢/Æcí2q*[ Ä‡ìrZqÛ¼¤›¼”›³ÔŠ ˜çÕu`Ø×åat×˜¡UG€Í_êGğÍ{ê\°Æí8²mJRHÂInI‰2*u ØçåutØØ¥eDÔÌ ªÀıoşSÿ‚?Şoç“õR8ÂmnRS‚BNw‹™X•PüÃş.?£¯Æí>2oªSÀÂ/îcó–:,ó¢:lı>rošSÔÂ ®gÃ•nÓ³â
6xéq6[©<Ÿ®Ã±n“¸’RzB\ÎF+ šgÔÕ` ×Çámw’YRELşJ?ˆïÙså4ôè¸±MKŠH˜ÉUi Ñ?ã¯öù>=/®cÃ–.#³¦
8üí~2_ªGÀÍoêSğÂ;îl³’
xò]zF\Í*} Şgç•uPØÃån4Ó¨¢FÚwäÙt¥„õ\¸Æm:RlÂR.BcV4Ÿ¨—ÁQoƒ“Ş'²eJTÈÀ©oÁï²3Êj(Ğác÷–9Q-¢~_½Î}kP—ƒÑ^#‡¦E6Lé
18ë­p‚[ŞD§Œ…ZÄö,¹"&zeÔö ¹'Í%jdĞÔ£à†7İ)fa°ñKûˆ¼™N°ø‹ıX¾EOŒËÚ(¤áD·Œ‰Zõ<¸îsºZÄú,¼â6{©6©7Á)o¡Ç²-JbHÖIa	¹1M+Š`˜×Õa`×—áQwƒ™^°ıKşH¿‰OÙå8´íH²IJIÉ9i-"s¦Züü¾>¯»ÃÌ®* şÿ½Î_ë‡ğ{Ö\¡½=NnK“ˆ’RuXşEŒßÚ'äåt´Øˆ¥YDÅ¬ú<şn?“¯Òâ~6_©Á=o®SÃ‚.c·–	Q9­>o¾SÏ‚+Ş`§—ÅQlÃ’.c²V
A8Ï­kÂP®CÃ.£´†9Vm²_ÊGèÍqj[Ä“ì’2jrPÚCäÎ4«¨€_ß‡çİufXÕ`ü×ş!§ŸÅWìÁr/šcÔÖ ¡'Ç¥mDÒL¢JHı	~y7Öia³±Jˆø™}U@÷ù[ı¾|×´¡H‡‰]YE=îz3œê0ñ+û ¼‡Îk¶P‰Ù>%/¤ãÄ¶,‰"&u%äõt¸ØeZTÄÀ¬¯Âî~3ŸªÀñoû“ü’>o²SÊB(Îak—‘SÓ‚"fw•Põøş=®_Ã‡îs¶Z	ù<½.c»–‘:¬ò:~lß’'ÒebTÖ¡zœıV>A/£ÛÆ$­$‚dT—€‘_Ó‡âvvY5<èî1s«š ”ÿĞ¿ãÏö+ù ½'Îek”Ğ£ÓÆ"-&beTñ »¿Ìêğô»øŒ½ZDûŒ¼šû°¼‹Î«µ@ˆÏÙkå´óÈº)Lá
7¸éMq
[¸ÄlšRÂp®[Ã„®ƒ¶	7¹)M!
g¸ÕM`ÊWèÁqo›“Ô’ ’gÒUb@ÖOá÷¸¹MM
JxÈİifQ°şÿ¸¿ÍOêKğÈ»éL±
¸ø}Z^DÇŒ­ZDşL¿ŠØûå|´Ş§¹EMÊz(Üáf7•)Pá÷¾9O­Âx®]C†N¶x‰Y6E)á:7¬éB1k»Œ“Ú$òdºTŒÀš/Ôãà¶7É)i!'³¥JÈü©~¿·ÏÉké±3Ëª(€á_÷‡ù]}^}}WAW[ß„§Ü…fÕ6 é'ñ%{¤Ü„¦…6é61)+¡ ‡§İEfLÕ
 øçıu~Xß…gÜÕf Õ'àåwôÙx¥DöL¹
8úm|Ò^"G¦ME
LøÊ=hîQsƒš÷°¹KÍªy@İæ{õ¸öy:],Æb-bq[±‹¼˜[°Ä‹ì˜²JpÈÛéd±‹°˜‹ÕX ÅGìÍr*Z`Ä×ì¡rš}TŞ@§Å[ìÄ²,ŠbÖua×µaH×‰aY…1\ë†0+Ö`¡Ç±mK’H’IRII>I/‰#Ù&%%$ää´´ˆˆ™YU üÿş?ÿ¯ÿÃÿî?ó¯úüş>?¯¯ÃÃî.3£ª ı?şoÿ“ÿÒ?âoöSù=>no““Ò"rfZUÀü¯şÿ¾?Ï¯ëÃğ®;Ã¬®¾~Ÿ»×Ì¡jıSşB?oÛ“ä’4’h’QRC‚NK·ˆ‰YY<üî>3¯ªÀş/ÿ£ÿÆ?í/òcúV<Á./££Æ-="nfS•şsÿš?Ôïà³÷Ê9hírsšZÄğ¬»Â®zœş?±/Ë£è†1]+†`Öqa—´‘H“‰Ru>Xï…sÜÚ&$å$´äˆ´™H•	Pùı>>o¯“ÃÒ."c¦V<ÿ®?Ã¯îó¾:¬ûÂ<®n“¾²{Ê\¨Æm?’oÒSâB6Ni‘8“­RB~N_‹‡ØeVTÁ ¯¿ÃÏî+ó ºÌıj>PïƒóŞ:'¬åB4Îh«‘@“Òât¶X‰Y<Å.,ã¢6