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
from data_generator_class import load_data_generator,load_data_generator_simple
from data_generator_class import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
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
train_txt = 'trainingClass.txt'
val_txt = 'validationClass.txt'
test_txt = 'testingClass.txt'

N_Class = 3
LOW_DIM = 1
idOar = Age
argvs = ['-i', '-u']
ssRatio = 1.0  # float(sys.argv[3])/100.0
PB_FLAG = "Class"  # to modify according to the task. A different evaluation function (test.py) will be used depending on the problem

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
        model_vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        for layer in model_vgg16.layers[:15]:
            layer.trainable = False

        x = Flatten(name='flatten')(model_vgg16.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        # x = BatchNormalization(name='bm1')(x)
        x = Dense(N_Class, activation='softmax', name='predictions')(x)
        model = Model(model_vgg16.input, x)

        rn = Input(shape=(N_Class,))
        weightedRn = merge([rn,x],mode='mul')
        modelRn = Model(input=[model_vgg16.input,rn], output=weightedRn)

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

        self.network.summary()
        self.networkRn.summary()

        
        self.networkRn.compile(optimizer=sgd,
                             loss='categorical_crossentropy',metrics=['accuracy'])
        self.network.compile(optimizer=sgd,
                             loss='categorical_crossentropy',metrics=['accuracy'])

        self.network.save_weights(self.fileWInit)
        self.network.save_weights(self.fileWbest)

        self.rni=np.ones((len(trainT),1),dtype=np.float)*np.ones((1,LOW_DIM))
        improve=True
        for it in range(itmax):
            if it == 0:
                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)

            else:
                f = open('test2_rni_objs.pkl','rb')
                self.rni = pickle.load(f)
                f.close()
                for i in range(self.rni.shape[0]):
                    if self.rni[i] < 0.001:
                        self.rni[i] = 0.001

                improve=self.M_step_network(ROOTPATH,trainT,valT,test_txt, learning_rate)
            if not improve:
                break

            
            (gen_training, N_train), (gen_test, N_test) = load_data_generator_List(ROOTPATH, trainT[:], test_txt)
            Ypred, Ytrue = extract_XY_generator(self.network, gen_training, N_train)
            Ntraining=int(validationRatio*N_train)
            
            # for iterEm in range(6):
            #     self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     if UNI:
            #         self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     self.E_step(Ypred,Ytrue)

            if DISPLAY_TEST:
                (gen_test, N_test) = load_data_generator_simple(ROOTPATH, test_txt)
                self.evaluate(gen_test, N_test, WIDTH)

    

    def custom_loss(rn):
        def loss(y_true, y_pred):
            # return K.mean(K.sum(rn*K.square(y_pred-y_true), axis=-1))
            return -K.mean(K.sum(rn*y_true*K.log(y_pred), axis=-1))
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
                  nesterov=True)

        self.network.load_weights(self.fileWInit)
        # self.networkRn.load_weights(self.fileWInit)

        self.networkRn.compile(optimizer=sgd,
                             loss='categorical_crossentropy',metrics=['accuracy'])
        self.network.compile(optimizer=sgd,
                             loss='categorical_crossentropy',metrics=['accuracy'])

        # self.network.compile(optimizer=sgd,loss=self.custom_loss(self.rni))

        # Replicate rni LOW_DIM times
        wei = np.repeat(wei, N_Class, axis=1)

        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei, valMode=VALMODE, validation=validationRatio, subsampling=ssRatio)

        history=self.networkRn.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoch,
                                             verbose=1,
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
    
        i=0
        Ypred=[]
        Y=[]
        for x,y in generator:
            if i>=n_predicuY≈5lË“1bkñPë”æ"¶{≈¨ˆ9>m/íc“V"A&O•ƒ¯¨ΩBN{ãúòñQ0√´Ó ≥ø Ë˚Ò|ªûó∫LÛä:Ïır8⁄md“T¢@ÜO›ÊxµHˆIy	96m)a2W™A@œèÎ€§ªƒå¨ö˛pøõœ‘´‡Ä∑ﬂ…gÈq0€´‰Ä¥ü»óÈQqõæè∞õÀ‘®†ÅGﬂçg⁄Ud¿‘Ø‡É˜ﬁ9g≠BpŒ[ÎÑ∞úã÷°5G®ÕAjOêÀ”Ë¢1Fkçös‘⁄ §Áƒµlà“buXÒ{º‹é&•4ÑËú±VÅ8ü≠W¬AnOìã“¢uFXÕj|–ﬁ#Á¶5E(Ã·j7êÈSÒ;ælèí“t¢XÜE]∆z-‚v6Y)!<ÁÆ5C®Œkøêè”€‚$∂dâô0ï+–‡£˜∆9m-brVZAœº´Œ ´ø¿èÔ€Û‰∫4åËö1TÎÄ∞üÀ◊Ë°qGõçTö@îœ–´„¿∂/…#È&1%+§‡Ñ∑‹âf50ËÎÒpªõÃî™ÄÛﬂ˙'¸Â~4ﬂ®ß¡Eoå”⁄"$Êdµàô{’†ˆ˘=}.^cáñQ6C©;ø¨è¬Ót≥òäX≈{Ï‹≤&
e8‘Ì`≤W Ahœëk”ê¢∆r-bt÷X°GºÕN*K†»áÈ]q[Ωé|õûó∞ëK”à¢Fu˙u|ÿﬁ%g§’D†ÃáÍpˆ[˘Ω<énì¥ííyR]F~MäwÿŸee‘†ª«Ã≠jP˛Cˇé?€Ø‰ÉÙû8ó≠QBCéNã¥òàïYP≈Ï˛2?™o¿”Ô‚3ˆj9Ì3Új:PÏ√Ú.:c¨÷!>gØïC–Œ#Î¶0Ö+‹‡¶7≈)l·7≤iJQ√πn∫r⁄z$‹‰¶4Ö(ú·V7Å)_°«ΩmNRKÇHûIWâY?Ö/‹„Ê65)(·!wßôEU¿˙/¸„˛6?©/¡#Ô¶3≈*,‡‚7ˆiy3∂j	˘3˝*>`ÔóÛ—z#úÊ51(Î°páõ›T¶@Ö‹˚Ê<µ.„πv:u,ÿ‚%vdŸ•0ÑÎ‹∞¶≈8¨ÌB2NjKê»ìÈR1kæPèÉ€ﬁ$ß§ÖDúÃñ* Ûß˙|¸ﬁ>'Ø•CƒŒ,´¢ Ü›Êwıxıxˆ]y]=n}ûröqT€Ä§üƒóÏërör⁄p§€ƒ§¨ÑÇûvô1U+Ä‡ü˜◊˘a}ûqWõÅTüÄóﬂ—g„ïvŸ3Â*4‡Ë∑ÒI{âô6)0·+˜†πGÕjzP‹√Ê.5#®Êu?òÔ’s‡⁄7‰Ètôï5PË√Òn;ì¨í~r_öG‘Õ`™W¿¡oÔìÛ“:"lÊR5h˛QÉüﬁÁ±uKò»ïiP—„æ6©;¡,Ø¢∆~-¢w∆Ym|Ú^:G¨ÕB*N`ÀóËëqSõÇûpóõ—T£ÄÜ›7ÊiuÛµz‹˘f=.p„õˆπ0ç+⁄`§◊ƒ°láíRvBYE;åÏö2Íp∞€À‰®¥ÅHüâWŸe?îÔ–≥„ 6(È!q'õ•TÑ¿úØ÷·>7Ø©C¡/ª£ÃÜ* ˆg˘}0ﬁkÁêµS»¬)naó≤Jsà⁄dı∏ç{⁄\§∆≠<ÇnS∑Ç	^yù=VnAè≤ t®ÿÅe_î«–≠c¬V.A#è¶≈4¨ËÇ1^káêùS÷B!gªïLê ËÚ1zkú–ñ#—&#•&Â<¥Ó≥πJ˙y|›&w•Dı∏˙|˙^<«Æ-C¢NKΩéy[ùñ|ë∑≤	Jy›9fmpÚ[˙DºÃé*†Ùá¯ù}V^AèΩ[ŒD´åÄö‘˜‡πwÕjuÿÛÂz4‹Ë¶1E+å‡ö7‘È`±À±hãëXìÖR¬v.Y#Ö&Â64È(±!KßàÖY\≈,˝">foï–Ú#˙f<’. „ßˆy<›.&c•Ò<ªÆÉ∫˜∫9LÌ
2xÍ]p∆[Ì≤|ä^«µmH“IbII1	+π ç'⁄ed‘‘††á«›mfRU@˛OˇãˇÿøÂOÙÀ¯®ΩANOããÿò•UD¿ÃØÍ˛;ˇ¨ø¬Ó{Ûú∫Ò:;¨ÏÇ2jwêŸSÂ4˛høëO”ã‚∂uI…5i(—!cßñQ<√Æ.£æΩ;Œl´í í“_‚GˆMy
]8∆mmRrBZNDÀå®öTˇÄøﬂœÁÎıp∏€Õd™TÄ¿üÔ◊Û·z7úÈV1+ø†è«€Ìd≤Tä@òœ’k‡–∑„…v)!5'®ÂAtœò´’@†œ«ÎÌp≤[ D®ÃÅjê˜”˘b=nqõ≤äpò€’d†‘á‡ùw÷YaºÒN;ã¨òÇ^p«õÌT≤@äOÿÀÂh¥—H£âF5:hÏ—r#öf’0†