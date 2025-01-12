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
     uzXÜÅf,Õ" ægõxğİ{æ\µı9~m’wÒYbELñ
;¸ìrZtÄØ¬¥BÎ|« —¿ÑOã‹ö¹5M(Êah×‘aS—‚^s‡šTö@¹Í;êl°Òâx¶]II=	.y#&e1ë°°‹ËØ¨¥ADÏŒ«Ú ¤ÿÄ¿ìòút¼Ø%[¤Ä„¬œ‚q7›©T Ÿ¿×Ïák÷¹SÍ*~`ß—çÑuc˜Öa0×«á@·É[é±<‹®ƒµ^Ç¹mMJrHÚIdÉ©0+ß §ÇÅmlÒR"BfNU€øŸıWşAŸÛ×ä¡t‡˜UV@Áï»óÌº*àú7üé~1«·À‰oÙå24êh°ÑKãˆ¶I5	(ù!}'eW”ÁP¯ƒÃŞ.'£¥FÍ<ªn Ó¿âö{ù½6i;‘,“¢r}^tÇ˜­UB@ÎOë‹ğ˜»ÕL Êèıq~[Ÿ„—Ü‘f•2êsğÚ;äì´²ŠyXİf|Õ ÷§ùE}Şz'œåV4Á(¯¡CÇ-[¢D†L
xñ{¶\‰=5.hã‘v™2*pàÛ÷ä¹tšuTØÀ¥oÄÓì¢2j}Şsçš5TèÀ±oË“è’1Rk‚PC×![§„…\œÆ-1"k¦P…Üş&?¥/Äãì¶2	*y İ'æeuØğ¥{ÄÜ¬¦>|ï3×ª!@çõ[øÄ½lR‚tX—…Q\Ã†.#¶f	90í+ò`ºWÌÁj/ãÓö"9&m%dòTº@ŒÏÚ+äà´·È‰iY3¼ê0û«ü€¾Ï·ëÉp©Á4¯¨ƒÁ^/‡£İF&M%
døÔ½`WÛdŸ”—Ğ‘cÓ–"&s¥ôü¸¾OºKÌÈª)@á÷»ùL½
xû|–^³½JHû‰|™7°éKñ»¹L
xôİx¦]ELı
>xïsÖZ!ç¼µNË¹hZs„Ú¤ö¹<.c´Ö¡9G­BzN\Ë†(!Vg_°ÇËíh²QJCˆÎkµˆóÙz%äö4¹(!Zg„Õ\ Æí=rnZS„Â®v™>/°ãËö(¹!M'ŠeXÔÅ`¬×Â!ng“•RÂsîZ3„ê°öù8½-NbK–H‘	S¹>zoœÓÖ"!&g¥DğÌ»ê°úüø¾=O®KÃˆ®Cµû¹|w´ÙH¥	Dù½:lû’<’nS²B
NxËh–QQƒ¾·»ÉL©
8ÿ­Â_îGóz\ôÆ8­-BbNVKŸ¹WÍjßÓqz[œÄ–,‘"¦r|ôŞ8§­EBLÎJ+ˆà™wÕ`õøñ}{\—†]3†jösù=4îh³‘JˆòzuØö%y$İ$¦d…œğ–;Ñ,£¢}=nw“™Rpş[ÿ„¿Üæõ4¸èqZ[„Äœ¬–>s¯šÔş ¿§ÏÅkìĞ²#Êf(Õ!`ç—õQxÃnS±¾x[ÖD¡‡ºLöJ9í9rmRtÂX®ECŒÎ+´àˆ·ÙIe	ù0½+Î`«—À‘oÓ“â6riQ4Ã¨®C¿Û»äŒ´š”ùP½Î~+Ÿ —ÇÑmc’VA2OªKÀÈ¯éCñ;»¬Œ‚t÷˜¹UM Êèßñgû•|Şç²5JhÈÑic‘±2ªx€İ_æGõxú]|Æ^-¢}F^MŠ}XŞEgŒÕZ ÄçìµrÚydİ¦p…Üô¦8…-\âF6M)
a8×­aBWA[„›Ü”¦…3Üê&0å+ôà¸·ÍIjIÉ3é*1 ë§ğ…{ÜÜ¦&%<äî4³¨ŠXÿ…Üßæ'õ%xäİt¦X…\üÆ>-/¢cÆV-"¦_Åìır>Zo„ÓÜ¢&e=îp³›Ê¨ğ{ßœ§Öa<×®!C§[¼Ä,›¢†pÖt¡‡µ]HÆIm	y2]*F`ÍêqpÛ›ä”´ˆ“ÙR%dşT¿€ßÛçäµtˆØ™eUÀğ¯ûÃü®>¯¾Ï¾+Ï «ÇÀ­oÂSîB3jô“ø’=RnBSBt›˜”•PÃÓî"3¦jüóş:?¬ïÂ3îj3êğò;úl¼Ò"{¦\…ı6>i/‘#Ó¦"&|å4÷¨¹AMŠ{ØÜ¥fÕ< îó½z\û†<.c±±8‹­X‚E^LÇŠ-XâEvLÙ
%8äít²XŠEXÌÅj,Ğâ#öf9-0âköP¹Í>*o ÓÇâ-vbYE1ëº0ŒëÚ0¤ëÄ°¬‹Â®uC˜Îk°Ğ‹ãØ¶%I$É$©$$Ÿ¤—Ä‘l“’rrZZDÄÌ¬ª şÿŸÿ×ÿá÷ŸùWı~ŸŸ××áaw—™QU€şÿ·ÿÉéñ7û©|··ÉIi	93­*`şWÿßŸç×õax×aVW_¿‡ÏİkæPµÈş)¡Ç·íIrII4É(©!A'¥[ÄÄ¬¬‚~wŸ™WÕ`ÿ—ÿÑãŸöù1}+`——ÑQcƒ–7³©Jÿ¹ÍêwğÙ{å´ö¹9M-
bxÖ]aW½N‹ŸØ—pº[ÌÄª,€âöwù}5h÷‘yS~q›·Ô‰`™Õ1`ë—ğ‘{Óœ¢q=®tƒ˜W°ÁKïˆ³ÙJ%äùt½u[˜Ä•lÒâr6ZiÑ<£®½>o»“Ì’*`òWúA|Ï+× ¡GÇmZRDÂL®JˆşµÈ÷éyq¶t‰™5U(Àáo÷“ùR=n~SŸ‚Şqg›•TÀ“ïÒ3âj6Péñ>;¯¬ƒÂ.w£™F0úküĞ¾#Ï¦+Å ¬çÂ5nhÓ‘b–rs´Ú¤ùD½zœô–8‘-S¢BN}x—QVC»·Ì‰jõ3øê=pî[ó„ºŒö94í(²aJWˆÁYo…Üò&:e,Ôâ ¶gÉi0Ñ+ã ¶É=i.Q#ƒ¦7¼éN1«¸€_ÚGäÍtªX€Å_ìÇò-zb\ÖF!'ºeLÔÊ ¨çÁuo˜ÓÕb Ögáw°ÙKå´ùH½	Ny8–mQC²N
K¸ÈiZQÃ¼®»¾ºÌôª8€í_òGúM|Ê^(Ç¡mG’MRJBHÎIk‰™3Õ* àç÷õyxİfvU õ?øïısşZ?„ïÜ³æ
58èíqr[šD”ÌªÀò/úcüÖ>!/§£ÅF,Í"*f`Õàñwû™|•÷³ùJ=îystñ»µLˆÊhõxóz\ñ;½,b–t‘“µRÂyn]†rvtÙ¥5DèÌ±jø“ıR>BoSÛ‚$d—”‘P“ƒÒ"w¦YEüú><ï®3Ãª. ã¿öù;ı,¾b–{Ñ£¶	=9.m#’fU2@êOğËûè¼±N‹¸˜UZ@ÄÏì«ò ºÌßê'ğå{ôÜ¸¦E:LìÊ2(êap×›áT·€‰_Ùå=tîX³…JÈö)y!'¶eIÉ0©+Á ¯§ÃÅn,Ó¢"f}p÷›ùT½ ÛŸä—ô‘x“RBq[»„Œœšñ0»«Ì€ªÀ÷ïùsı>tï˜³ÕJ ÈçéuqÛµdˆÔ™`•Ğñcû–<‘.£²
}8Şmg’UR@ÂOîKóˆºLõ
8øí}r^ZG„Í\ªF Í?êoğÓûâ<¶n	¹2*z`Ü×æ!u'˜åUtÀØ¯åCôÎ8«­@‚OŞKçˆµYHÅ	lù=2njSÂîr3šjĞğ£ûÆ<­.c¾V;ß¬§Ân|Ó"¦qEŒôš8”íP²CÊN(Ë¡h‡‘]S†Bv{™•6é3ñ*; ì‡òzv\Ùr:ZlÄÒ,¢bV}·ŸÉWéq?›¯Ôƒà7×©aA±[Ë„¨œV7ß©gÁo°ÓËâ(¶aI‰1Y+… œçÖ5a(×¡aG—QZC„Î«¶ ‰?Ù/å#ôæ8µ-HâIvI	59(í!rgšUTÀÀ¯ïÃóî:3¬ê0şkÿ¿ÓÏâ+ö`¹Í1jkĞ“ãÒ6"i&Q%¤ş¿¼Îë´°ˆ‹ÙX¥DüÌ¾* ûÇü­~_¾GÏkÚP¤ÃÄ®,ƒ¢w½Nu˜ø•}PŞCç5[¨ÄlŸ’Òqb[–D‘“ºòz:\ìÆ2-*b`ÖWáw¿™OÕàø·ıI~I‰7Ù)e!ç°µKÈÈ©iA³»Ê¨ú|ÿ?×¯áC÷9[­‚|^‡±]K†H	Vy?¶oÉé21*k Ğ‡ãİv&Y%$üä¾4¨›ÁT¯€ƒßŞ'ç¥uDØÌ¥jĞü£ş?½/Îcë–0‘+Ó ¢Æ}mRw‚Y^EŒıZ>DïŒ³Ú
$øä½tX›…TœÀ–/Ñ#ã¦6)<á.7£©F?ºoÌÓê"0ækõ¸óÍz*\àÆ7í)raW´ÁH¯‰CÙ%;¤ì„²ŠvÙ5e(Ôá`·—ÉQi‘>¯²Ê~(ß¡gÇ•mPÒCâN6K©9_­Â}n^S‡‚^vG™U:@ìÏò+ú`¼×Î!k§…SÜÂ&.e#”æµ3Èê)pá÷´¹H	Zyİ<¦n¼ò:{¬Ü‚&e7”éP±Ë¾(¡[Ç„­\‚FM7ŠiXÑc¼Ö!;§¬…BÎv+™ •'ĞåcôÖ8¡-G¢MFJMÊyhİfs•ôóøº=LîJ3ˆêpõøô½x][†D–zó¶:	,ù"=&ne”òºsÌÚ*$àä·ô‰x™U6@éñ;û¬¼‚{·œ‰V5?¨