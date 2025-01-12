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
                                             nb_epoch=nbEpoc≠;¬lÆRÇ~_∑á…]iQ=Æ~üæœ±kÀê®ì¡R/ÇcﬁV'Å%_§«ƒ≠lÇRBwéY[Öú¸ñ>/≥£ (˝!~güïW–¡cÔñ3—*#†Êı=xÓ]sÜZˆ|π7∫iL—
#∏Êu:XÏ≈r,⁄b$÷d°á∞ùK÷H°	GπM:Jl»“)baW±KøàèŸ[Â¥¸àæOµ»¯©}AO∑ã…X©A<œÆ+√†Æ√ΩnSªÇûzúÒV;Å,ü¢∆qmítíXíERL¬J.H„âv55(Ë·qwõôTï êˇ”ˇ‚?ˆo˘˝2>joê””‚"6fi0Û´˙ ºˇŒ?ÎØÉ˚ﬁ<ßÆCºŒ+ª†åá⁄dˆTπ ç?⁄o‰”Ù¢8Üm]FrMJt»ÿ©eAœ∞´À¿®Ø¡CÔé3€™$Ä‰üÙó¯ë}SûBéq[õÑîúêñ—2#™f ’?‡Ô˜Û˘z=Óv3ô* Á˚ı|∏ﬁg∫UL¿ /Ë„Òv;ô,ï"Êsı8ÙÌx≤]JFHÕ	jy›3Êj5ËÛÒz;úÏñ2*s†⁄‰˝tæXèÖ[‹ƒ¶,Ö"Êv5(ı!xÁùuVX¡oº”Œ"+¶`Ö‹Òf;ï,ê‚ˆr9m4“h¢QFCç{¥‹à¶E5Ë˙1|Îû0ó´—@£è∆Ì4≤häQX√Ön”∂"	&y%$ˆdπç0ök‘–†£«∆-m"RfBU@˚è¸õ˛ø∞èÀ€Ë§±DãåòöT¿ªÔÃ≥Í
0¯Î˝pæ[œÑ´‹Ä¶≈7ÏÈr1k¥–à£ŸF%$˙dº‘é õß‘Ö`ú◊÷!a'ó•QD√åÆ¥˛øπOÕÍx∞›KÊHµ	H˘	}9m7íiRQCæNãªÿå•Zƒ¸¨ææ{œú´÷ °?«ØÌCÚN:K¨»Ç)^aóΩQNCãéõµTà¿ôo’‡Ú7˙i|—#∑¶	E9Ì:2lÍR0¬kÓP≥É (˜°yGùVzAœ∂+… ©'¡%o§”ƒ¢,Übvqµ4àËôqUÄÙü¯ó˝Q~Cüé€±dãîòêïS–¬#Óf3ï*‡Û˜˙9|Ì2w™Y@≈Ï˚Ú<∫n”∫"Êz5Ëˆ1y+ù ñg—c∞÷·8∑≠IBII;â,ô"&pÂÙÙ∏∏çMZJD»Ã©jˇ≥ˇ ?ËÔÒs˚ö<îÓ≥≥ 
(¯·}wûYWÖ\ˇÜ?›/Êcı8Ò-{¢\ÜF6zi—6#©&%?§Ôƒ≥Ïä2Íupÿ€Âd¥‘à†ôG’`/≠#¬f.U#ÄÊı7¯È}q[∑Ñâ\ô=0ÓkÛê∫ÃÚ*:`Ï◊Ú!zgú’V ¡'Ô•sƒ⁄,§‚∂|â7µ)H·	wπM5
h¯—}cûVÅ1_´á¿ùo÷S·7æiOë”∏¢FzM v(Ÿ!e'îÂP¥√»Æ)C°ªΩLéJàÙôxïPˆC˘=;ÆlÉíw≤YJEÃ˘j=ÓsÛö:Ï≤; l®“bñ_—„ΩvY;Ö,ú‚6q)°4á®ùAVOÅﬂ∏ßÕEjL– #ËÊ1u+ò‡ïw–ŸcÂ4Ò(ª°LáäXˆEy›:&lÂ4Úh∫QL√ä.„µvŸ9e-‚p∂[…©<Å.£∑∆	m9m2RjBPŒCÎé0õ´‘Ä†ü«◊ÌarWöATœÄ´ﬂ¿ßÔ≈sÏ⁄2$Íd∞‘ã‡ò∑’I`…È1q+õ†îá–ùc÷V!'ø•OƒÀÏ®≤JàﬂŸgÂtÿªÂL¥ ®˘A}û{◊ú°VÅ=_ÆG√çnS¥¬ÆyCù{±ã∂â5Y(≈!lÁí5Rh¬QnCìé≤täXò≈Ul¿“/‚cˆV9-?¢o∆SÌ2~j_ê«”Ìb2VjAœ≥Î 0®Î¡pØõ√‘Æ Éßﬁgº’N ÀßËÖq\€Ü$ù$ñdëì∞í“x¢]FFM
zx‹›f&U% ‰ˇÙø¯è˝[˛Døåè⁄‰Ù¥∏àçYZEÃ¸™> ÔøÛœ˙+¸‡æ7œ©k¡Ø≥√ .(„°vô=U.@„èˆ˘4Ω(éa[óÑë\ìÜ2vjY≈3ÏÍ20Ík–ª„Ã∂*	 ˘'˝%~dﬂîß–Öc‹÷&!%'§ÂD¥Ãà™@ı¯˚˝|æ^áª›L¶J¸˘~=Æw√ôn∞Ú˙xº›N&K•Ñ˘\Ω};ûlóíRsÇZD˜åπZ˙|ºﬁ'ª•LÑ ®ˆy?ù/÷c·7±)K°áπ]MJ}ﬁygùVp¡Ô¥≥»ä)X·wºŸN%§¯ÑΩ\éFç4öhî—P£É∆-7¢iFQ∫~ﬂ∫'ÃÂj4–Ë£ÒF;ç,öb÷p°«¥≠HÇI^Iâ=Y.E#åÊ54ËË±qKõàîôPï–˛#ˇ¶?≈/Ï„Ú6:i,—"#¶f<Ó;Û¨∫˛z?úÔ÷3·*7†ÈGÒ{∫\å∆-4‚h∂QIâ>/µ#»Ê)u!ÁµuHÿ…ei—0£´∆ ≠?¬oÓSÛÇ:l˜í9RmR~B_éG€çdöTî¿êØ”√‚.6c©1?´Ø¿ÉÔﬁ3Á™5@.m#ífU2@ÍOÀ˚Ëº±Nã∏òçUZ@ƒœÏ´Ú ∫ÃﬂÍ'Â{Ù‹∏¶E:LÏ 2(Íap◊õ·T∑Äâ_ŸÂ=tÓX≥ÖJ»ˆ)y!'∂eI…0©+¡ Øß√≈n,”¢"f}p˜õ˘TΩ é€ü‰óÙëxìùRBq[ªÑåúöÒ0ª´ÃÄ™¿˜Ô˘s˝>tÔò≥’J »ÁÈuq€µdà‘ô`ï–Òc˚ñ<ë.£≤
}8ﬁmgíUR@¬OÓKÛà∫Lı
8¯Ì}r^ZGÑÕ\™F Õ?Ío”˚‚<∂n	π2*z`‹◊Ê!u'òÂUt¿ÿØÂCÙŒ8´≠@ÇOﬁKÁàµYH≈	l˘=2njSê¬Ór3öj–£˚∆<≠.cæVÅ;ﬂ¨ß¬n|”û"¶qEåÙö8îÌP≤C N(À°háë]SÜBv{ôï6È3Ò*;†ÏáÚzv\Ÿ%=$Ód≥îäòÛ’z ‹ÁÊ5u(ÿ·ewîŸP•ƒ˛,ø¢∆{Ì≤v
Y8≈-l‚R6BiQ;É¨ûæqOõã‘ò†ïG–ÕcÍV0¡+Ô†≥« -h‚QvCô;∞ÏãÚ∫uLÿ %h‰—t£òÜ]0∆kÌ≤s Z(ƒ·l∑í	Ry]>Foç⁄r$⁄d§‘Ñ†úá÷a6W©A?èØ€√‰Æ4É®ûWøÅOﬂãÁÿµeH‘…`©¡1o´ì¿í/“c‚V6A)°;«¨≠BN~KüàóŸQeî˛ø≥œ +Ë‡±wÀôhïPÛÉ˙<˜Æ9C≠{æ\èÜ›4¶hÖ\ÛÜ:,ˆb9m1k≤PäCÿŒ%k§–Ñ£‹Ü&%6dÈ±0ã´ÿÄ•_ƒ«Ï≠rZ~Dﬂåß⁄d¸‘æ èß€≈d¨‘Ç ûg◊ïaP◊É·^7á©]AOΩŒx´ù@ñO—„∏∂I:I,…")&a%§ÒDªååöÙ∏ªÕL™J »ˇÈÒ˚∑¸â~µ7»Èiq≥¥äò˘U} ﬁÁüıW¯¡}oûS◊Ç!^gáï]P∆CÌ2{™\Ä∆Ì7ÚizQ√∂.	#π&%:dÏ‘≤ ägÿ’e`‘◊‡°w«ômU@ÚO˙K¸»æ)O°«∏≠MBJNHÀâhôU3ÄÍ˜˚˘|ΩwªôLï
¯Û˝z>\ÔÜ3›*&`ÂÙÒxªùLñJÛπz˙v<Ÿ.%#§Êµ<àÓsµÙ