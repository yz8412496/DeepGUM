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

        self.networkRn.compile(optimizer=rmsprop, loss={'classification': self.custom_lossuYÅ5lèÒ1bk–P‘Ó¾"¦{Å¬ö9>m/’cÒV"A&O¥Äø¬½BN{‹œ˜–Q0Ã«î ³¿Êèûñ|»—ºLóŠ:ìõr8ÚmdÒT¢@†OİæxµHöIy	96m)a2WªA@ÏëÛğ¤»ÄŒ¬šşp¿›ÏÔ«à€·ßÉgéq0Û«ä€´ŸÈ—éQq›¾°›ËÔ¨ GßgÚUdÀÔ¯àƒ÷Ş9g­BpÎ[ë„°œ‹Ö¡5G¨ÍAjOËÓè¢1FkšsÔÚ ¤çÄµlˆÒbuXñ{¼Ü&¥4„èœ±V8Ÿ­WÂAnO“‹Ò¢uFXÍj|ĞŞ#ç¦5E(Ìáj7éSñ;¾l’Òt¢X†E]Æz-âv6Y)!<ç®5C¨Îk¿ÓÛâ$¶d‰™0•+Ğà£÷Æ9m-brVZAÏ¼«Î «¿ÀïÛóäº4Œèš1Të€°ŸË×è¡qG›Tš@”ÏĞ«ãÀ¶/É#é&1%+¤à„·Ü‰f50èëñp»›Ì”ª€óßú'üå~4ß¨§ÁEoŒÓÚ"$ædµˆğ™{Õ öù=}.^c‡–Q6C©;¿¬Âît³˜ŠXğÅ{ìÜ²&
e8Ôí`²WÊAhÏ‘kÓ¢Ær-btÖX¡G¼ÍN*K È‡é]q[½|›—°‘KÓˆ¢Fuúu|ØŞ%g¤ÕD Ì‡êpö[ù½<n“´’’yR]F~MŠwØÙeeÔğ »ÇÌ­jPşCÿ?Û¯äƒô8—­QBCN‹´˜ˆ•YPÅìş2?ªoÀÓïâ3öj9í3òj:PìÃò.:c¬Ö!>g¯•CĞÎ#ë¦0…+Üà¦7Å)lá7²iJQÃ¹nºrÚz$Üä¦4…(œáV7)_¡Ç½mNRK‚HIW‰Y?…/Üãæ65)(á!w§™EUÀú/üãş6?©/Á#ï¦3Å*,àâ7öiy3¶j	ù3ı*>`ï—óÑz#œæ51(ë¡p‡›İT¦@…Üûæ<µ.ã¹v:u,Øâ%vdÙ¥0„ëÜ°¦Å8¬íB2NjKÈ“éR1k¾PƒÛŞ$§¤…DœÌ–* ó§ú|üŞ>'¯¥CÄÎ,«¢ †İæwõxõxö]y]=n}ršqTÛ€¤ŸÄ—ì‘ršrÚp¤ÛÄ¤¬„‚v™1U+€àŸ÷×ùa}qW›TŸ€—ßÑgã•vÙ3å*4àè·ñI{‰™6)0á+÷ ¹GÍjzPÜÃæ.5#¨æu?˜ïÕsàÚ7äét™•5PèÃñn;“¬’~r_šGÔÍ`ªWÀÁoï“óÒ:"læR5hşQƒŸŞç±uK˜È•iPÑã¾6©;Á,¯¢Æ~-¢wÆYm|ò^:G¬ÍB*N`Ë—è‘qS›‚p—›ÑT£€†İ7æiuóµzÜùf=.pã›ö¹0+Ú`¤×Ä¡l‡’RvBYE;Œìš2êp°ÛËä¨´HŸ‰WÙe?”ïĞ³ãÊ6(é!q'›¥T„Àœ¯Öá>7¯©CÁ/»£Ì†* ögù}0ŞkçµSÈÂ)na—²JsˆÚdõ¸ğ{Ú\¤Æ­<‚nS·‚	^y=VnA²Êt¨Øe_”ÇĞ­cÂV.A#¦Å4¬è‚1^k‡SÖB!g»•LÊèò1zkœĞ–#Ñ&#¥&å<´î³¹Júy|İ&w¥Dõ¸ú|ú^<Ç®-C¢NK½y[–|‘·²	Jyİ9fmpò[úD¼Ì* ô‡ø}V^A½[ÎD«Œ€šÔ÷à¹wÍjuØóåz4Üè¦1E+Œàš7Ôé`±Ë±h‹‘X“…RÂv.Y#…&å64é(±!K§ˆ…Y\Å,ı">fo•Ğò#úf<Õ. ã§öy<İ.&c¥ñ<»®ƒº÷º9Lí
2xê]pÆ[í²|Š^ÇµmHÒIbII1	+¹ 'ÚedÔÔ  ‡ÇİmfRU@şOÿ‹ÿØ¿åOôËø¨½ANO‹‹Ø˜¥UDÀÌ¯êğş;ÿ¬¿Âî{óœºñ:;¬ì‚2jwÙSå4şh¿‘OÓ‹â¶uIÉ5i(Ñ!c§–Q<Ã®.£¾½;Îl«’ ’Ò_âGöMy
]8ÆmmRrBZNDËŒ¨šTÿ€¿ßÏçëõp¸ÛÍdªT€ÀŸï×óáz7œéV1+¿ ÇÛíd²TŠ@˜ÏÕkàĞ·ãÉv)!5'¨åAtÏ˜«Õ@ ÏÇëíp²[ÊD¨Ìj÷Óùb=nq›²Šp˜ÛÕd Ô‡àwÖYa¼ñN;‹¬˜‚^pÇ›íT²@ŠOØËåh´ÑH£‰F5:hìÑr#šfÕ0 ëÇğ­{Â\®F>o´ÓÈ¢)FaºqLÛŠ$˜ä•tØ“åR4Âh®QCƒ·´‰H™	U9 í?òoúSüÂ>.o£“Æ-2bjVPÁï¾3Ïª+Àà¯÷Ãùn=®rš~ß°§ËÅh¬ÑB#f•4è“ñR;‚lR‚q^[‡„\–F3ºjĞú#üæ>5/¨ãÁv/™#Õ& å'ôåxv55(èáqw›™T• ÿÓÿâ?öoùı2>joÓÓâ"6fi0ó«ú ¼ÿÎ?ë¯ğƒûŞ<§®C¼Î+» Œ‡ÚdöT¹ ?ÚoäÓô¢8†m]FrMJtÈØ©eAÏ°«ËÀ¨¯ÁCï3Ûª$€äŸô—ø‘}SBq[›„”œ–Ñ2#ªf Õ?àï÷óùz=îv3™* ğçûõ|¸ŞgºULÀÊ/èãñv;™,•"æsõ8ôíx²]JFHÍ	jyİ3æj5èóñz;œì–2*s Úäıt¾X…[ÜÄ¦,…"æv5(õ!xçuVXÁo¼ÓÎ"+¦`…Üñf;•,âör9m4Òh¢QFC{´Üˆ¦E5èú1|ë0—«Ñ@£Æí4²hŠQXÃ…nÓ¶"	&y%$öd¹0škÔĞ £ÇÆ-m"RfBU@ûü›ş¿°ËÛè¤±D‹Œ˜šTğÀ»ïÌ³ê
0øëıp¾[Ï„«Ü€¦Å7ìér1k´Ğˆ£ÙF%$úd¼Ô ›§Ô…`œ×Ö!a'—¥QDÃŒ®´ş¿¹OÍêx°İKæHµ	Hù	}9m7’iRQC¾N‹»ØŒ¥ZÄü¬¾¾{Ïœ«Ö ¡?Ç¯íCòN:K¬È‚)^a—½QNC‹›µTˆÀ™oÕàò7úi|Ñ#·¦	E9í:2lêR0ÂkîP³ƒÊ(÷¡yGVzAÏ¶+É ©'Á%o¤ÓÄ¢,†bvqµ4ˆè™qU€ôŸø—ıQ~CŸÛ±d‹”˜•SĞÂ#îf3•*àó÷ú9|í2wªY@Åìûò<ºnÓº"æz5èö1y+ –gÑc°Öá8·­IBII;‰,™"&påôô¸¸MZJDÈÌ©jÿ³ÿÊ?èïñsûš<”î³³Ê
(øá}wYW…\ÿ†?İ/æcõ8ñ-{¢\†F6ziÑ6#©&%?¤ïÄ³ìŠ2êupØÛåd´Ôˆ ™GÕ`úWüÁ~/Ÿ£×Æ!m'’eRTÂ@®OÃ‹î³µJÈùi}s·š	Tù ½?Îoë“ğ’;Òl¢RB}^{‡œVA1«»ÀŒ¯Úäş4¿¨Á[ï„³ÜŠ&å5tèØ±eK”È©SÁ/¾cÏ–+Ñ £§Æm<Òn"S¦B|û<—®C³
¸ôxš]TÆ@­Â{î\³†
8ömy]2FjMÊsèÚ1dë”°‹ÓØ¢%FdÍªp€Ûßä§ô…xœİV&A%¤ûÄ¼¬¾t˜›ÕT À‡ïİsæZwÙe5èğ±{Ëœ¨–Q?ƒ¯Şç¾5O¨ËÁh¯‘CÓ"¦t…œõV8Á-o¢SÆB-b{–\‘½2j{Ü“æ52hêQpÃ›î³°ŠØø¥}DŞL§ŠXüÅ~,ß¢'ÆemÒp¢[ÆD­‚z\÷†9]-b}^q›½T@›Ô›à”·Ğ‰cÙ%1$ë¤°„‹Ü˜¦E0Ìëê0°ëËğ¨»ÁL¯ŠØş%¤ßÄ§ì…rÚv$Ù$¥$„äœ´–‘9S­~~_Ÿ‡×İafW•PÿƒÿŞ?ç¯õCøÎ=k®PƒƒŞ'·¥IDÉ©:,ÿ¢?Æoíòr:ZlÄÒ,¢bV}·ŸÉWéq?›¯Ôƒà7×©aA±[Ë„¨œV7ß©gÁo°ÓËâ(¶aI‰1Y+… œçÖ5a(×¡aG—QZC„Î«¶ ‰?Ù/å#ôæ8µ-HâIvI	59(í!rgšUTÀÀ¯ïÃóî:3¬ê0şkÿ¿ÓÏâ+ö`¹Í1jkĞ“ãÒ6"i&Q%¤ş¿¼Îë´°ˆ‹ÙX¥DüÌ¾* ûÇü­~_¾GÏkÚP¤ÃÄ®,ƒ¢w½Nu˜ø•}PŞCç5[¨ÄlŸ’Òqb[–D‘“ºòz:\ìÆ2-*b`ÖWáw¿™OÕàø·ıI~I‰7Ù)e!ç°µKÈÈ©iA³»Ê¨ú|ÿ?×¯áC÷9[­‚|^‡±]K†H	Vy?¶oÉé21*k Ğ‡ãİv&Y%$üä¾4¨›ÁT¯€ƒßŞ'ç¥uDØÌ¥jĞü£ş?½/Îcë–0‘+Ó ¢Æ}mRw‚Y^EŒıZ>DïŒ³Ú
$øä½tX›…TœÀ–/Ñ#ã¦6)<á.7£©F?ºoÌÓê"0ækõ¸óÍz*\àÆ7í)raW´ÁH¯‰CÙ%;¤ì„²ŠvÙ5e(Ôá`·—ÉQi‘>¯²Ê~(ß¡gÇ•mPÒCâN6K©9_­Â}n^S‡‚^vG™U:@ìÏò+ú`¼×Î!k§…SÜÂ&.e#”æµ3Èê)pá÷´¹H	Zyİ<¦n¼ò:{¬Ü‚&e7”éP±Ë¾(¡[Ç„­\‚FM7ŠiXÑc¼Ö!;§¬…BÎv+™ •'ĞåcôÖ8¡-G¢MFJMÊyhİfs•ôóøº=LîJ3ˆêpõøô½x][†D–zó¶:	,ù"=&ne”òºsÌÚ*$àä·ô‰x™U6@éñ;û¬¼‚{·œ‰V5?¨ïÁsïš3Ôê °çËgÙe0Ôëà°·ËÉh©A3ªÀô¯øƒı^>G¯CÚN$Ë¤¨„\Ÿ†İ1fk•óÓú"<æn5¨òzœßÖ'á%w¤ÙD¥„ú¼ö9;­,‚bVw_µÈıi~Qƒ·Ş	g¹M0ÊkèĞ±cË–(‘!S§‚^|Ç