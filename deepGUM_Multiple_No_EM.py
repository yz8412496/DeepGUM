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
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from VGG16_rn import VGG16, extract_XY_generator
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
Age = "O" # 'Y', 'M', 'O' and 'All'
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
testing_txt = 'testAnnotations.txt'
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
ITER = 6
# ITER = 2
WIDTH = 224
BATCH_SIZE = 128
# BATCH_SIZE = 1
NB_EPOCH = 15
# NB_EPOCH = 50
PATIENCE = 1
# PATIENCE = 5
NB_EPOCH_MAX = 50
LEARNING_RATE = 1e-04
# validationRatio = 0.80
validationRatio = 1.0
fileWInit = ROOTPATH+"Forward_Uni_"+PB_FLAG+"_"+idOar+"_weights.hdf5"


class MixtureModel:
    ''' Class of forward model'''

    def __init__(self):

        self.network,self.networkRn = VGG16(LOW_DIM,weights='imagenet')

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
        
        layer_nb=16 # number of finetunned layer
        # train only some layers
        for layer in self.network.layers[:layer_nb]:
            layer.trainable = False
        for layer in self.network.layers[layer_nb:]:
            layer.trainable = True
        self.network.layers[-1].trainable = True

        # compile the model
        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        self.network.summary()
        # self.networkRn.summary()

        
        # self.networkRn.compile(optimizer=sgd,
        #                        loss='mse')
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
            
            # for iterEm in range(6):
            #     # self.E_step(Ypred,Ytrue)
            #     self.M_step_lambda(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     # self.M_step_pi(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     if UNI:
            #         self.M_step_U(Ypred[:Ntraining],Ytrue[:Ntraining])
            #     self.E_step(Ypred,Ytrue)

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
        
        # self.networkRn.compile(optimizer=sgd,
        #                        loss='mse')
        self.network.compile(optimizer=sgd,
                             loss='mse')

        # self.networkRn = self.network
        
        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei,valMode=VALMODE, validation=validationRatio,subsampling=ssRatio)

        history=self.network.fit_generator(gen_training,
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
            self.network.load_weights(self.fileW)
            # self.networkRn.save_weights(self.fileWInit)
            self.network.save_weights(self.fileWbest)
            return True
        else:
            self.network.load_weights(self.fileWbest)
            return False
            # return True



    
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
            if i>=n_predict:
                break
            pred=self.network.predict_on_batch(x)
            Ypred.extend(pred)
            Y.extend(y)
            i+=len(y)
        
        return np.asarray(Ypred), np.asarray(Y)

   
    def evaluate(self, generator, n_eval, l=WIDTH, pbFlag=PB_FLAG):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            generator: input a generşq›ŸÔ—à‘wÓ™bpñû´¼ˆ[µˆü™~°÷Ëùh½Ns‹š”õP¸ÃÍn*S Âî}sZ„ñ\»†:lñ;²lŠRÂunXÓ…bÖv!'µ%HäÉt©5_¨ÇÁmo’SÒB"NfK•ùSı>~oŸ“×Ò!bg–UQ Ã¿îó»ú¼ú<û®<ƒ®·¾	O¹Í8ªm@ÒOâKöH¹	M9
m8ÒmbRVBAO»‹Ì˜ª@ğÏûëü°¾Ï¸«Í@ªOÀËïè³ñJ;ˆì™rpôÛø¤½DL›Š˜ğ•{ĞÜ£æ5=(îas—šTó€ºÌ÷ê9píòtºXŒÅZ,Äâ,¶b	y1+¶`‰Ù1e+”à·ÓÉb)a1«±@‹Ø›åT´Àˆ¯ÙCå4û¨¼N‹·Ø‰eYÅ0¬ëÂ0®kÃ®Ã².
c¸Öa:W¬ÁB/cÛ–$‘$“¤’’|’^G²MJJHÈÉii3³ª
 øÿış_ÿ‡ÿİæ_õøı}~^_‡‡İ]fFU úüßş'ÿ¥Äßì§òz|ÜŞ&'¥%DäÌ´ª€ù_ış}_×‡á]w†Y]|ı>w¯™CÕ û§ü…~ß¶'É%i$Ñ$£¤†<–n³²

xøİ}f^U€ı_şGÿÚ_äÇô­x‚]^FGZzDÜÌ¦* üçş5¨ßÁgï•sĞÚ#äæ4µ(ˆáYw…\õ8ı-~b_–GÑcºVÁ:/¬ãÂ6.i#‘&¥2ê|°Şç¸µMHÊIhÉi3‘* òú}|Ş^'‡¥]DÆL­
xş]†_İæ}uX÷…y\İ&}%d÷”¹PÚ~$ß¤§Ä…lœÒ"q&[¥„üœ¾±;Ë¬¨‚^‡ŸİWæAu˜ûÕ| Şç½uNXË…hœÑV#&¥7Äél±²xŠ]XÆEmÒz"\æF5(úa|×!W§E_ŒÇÚ-dâT¶@‰Ù;å,´â¶yI	6y)!6g©A0Ï«ëÀ°¯ËÃè®1C« ›¿Ôà›÷Ô¹`ÚqdÛ”¤„“Ü’&e2Tê@°ÏËëè°±KËˆ¨™AU€ûßü§ş¼ßÎ'ë¥p„ÛÜ¤¦…<œî3±* ø‡ı]~F_Ú}dŞT§€…_ÜÇæ-u"XæEuØú%|äŞ4§¨…A\Ï†+İ ¦gÅlğÒ;âl¶R	y>]/†cİ&q%¤ô„¸œVA4Ï¨«Á@¯ÃÛî$³¤Š˜ü•~ß³çÊ5hèÑqc›–ş}_×‡á]w†Y]|ı>w¯™CÕ û§ü…~ß¶'É%i$Ñ$£¤†<–n³²

xøİ}f^U€ı_şGÿÚ_äÇô­x‚]^FGZzDÜÌ¦* üçş5¨ßÁgï•sĞÚ#äæ4µ(ˆáYw…\õ8ı-~b_–GÑcºVÁ:/¬ãÂ6.i#‘&¥2ê|°Şç¸µMHÊIhÉi3‘* òú}|Ş^'‡¥]DÆL­
xş]†_İæ}uX÷…y\İ&}%d÷”¹PÚ~$ß¤§Ä…lœÒ"q&[¥„üœ¾±;Ë¬¨‚^‡ŸİWæAu˜ûÕ| Şç½uNXË…hœÑV#&¥7Äél±²xŠ]XÆEmÒz"\æF5(úa|×!W§E_ŒÇÚ-dâT¶@‰Ù;å,´â¶yI	6y)!6g©A0Ï«ëÀ°¯ËÃè®1C« ›¿Ôà›÷Ô¹`ÚqdÛ”¤„“Ü’&e2Tê@°ÏËëè°±KËˆ¨™AU€ûßü§ş¼ßÎ'ë¥p„ÛÜ¤¦…<œî3±* ø‡ı]~F_Ú}dŞT§€…_ÜÇæ-u"XæEuØú%|äŞ4§¨…A\Ï†+İ ¦gÅlğÒ;âl¶R	y>]/†cİ&q%¤ô„¸œVA4Ï¨«Á@¯ÃÛî$³¤Š˜ü•~ß³çÊ5hèÑqc›–‘0“«Ò ¢Æ_íò}z^\Ç†-]"FfM
pøÛıd¾T€›ßÔ§à…wÜÙf%$ğä»ôŒ¸šTú@¼ÏÎ+ë °‡Ëİh¦QEŒş?´ïÈ³éJ1ë¹pÚt¤Ø„¥\„Æ­6i>Q/ƒ£Ş'½%NdË”¨Sß‚'Şeg”ÕP ÃÇî-s¢ZDı¾zœûÖ<¡.£½FM;Šl˜ÒbpÖ[á·¼‰Nµ8ˆíYrELôÊ8¨íArOšKÔÈ ©GÁoºSÌÂ*.`ã—öy3*`ñû±|‹—µQHÃ‰nµ2êypİætµˆõYxÅlöR9m>Ro‚SŞB'e[”Ä¬“Â.rcšVÁ0¯«ÃÀ®/Ã£î3½*`û—ü‘~Ÿ²ÊqhÛ‘d“”’’sÒZ"DæLµ
øù}}^w‡™]U@ış{ÿœ¿Öá;÷¬¹Bz{œÜ–&%3¤ê°ü‹ş¿µOÈËéh±K³ˆŠXõxüİ~&_¥Äıl¾R‚{Ş\§†]<Æn-¢rZ}Ş|§W¼ÁN/‹£Ø†%]$Æd­‚p[×„¡\‡†>}/c×–!Q'ƒ¥^Ç¼­NK¾H‰[Ù¥<„î³¶
	8ù-}"^fG•PúCüÎ>+¯ ƒÇŞ-g¢UF@Íê{ğÜ»æµ:ìùr=ntÓ˜¢FpÍêt°Ø‹åX´ÅH¬ÉB)a;—¬‘Bršt”Ø¥SÄÂ,®b–~³·Ê	hù}3jñSû‚<n“±R‚x]W†A]†{İ¦v<õ.8ã­vY>E/ŒãÚ6$é$±$‹¤˜„•\Æí22jjPĞÃãî63©* ÿ§ÿÅìßò'úe|ÔŞ §§ÅElÌÒ*"`æWõxÿÖ_á÷½yN]†xVvAµ;Èì©r´ßÈ§éEqÛº$Œäš4”è±SË‚(aW—Q_ƒ‡Şg¶UI É?é/ñ#û¦<….ã¶6	)9!-'¢eFTÍ ªÀßïçóõz8Üíf2U*@àÏ÷ëùp½Ît«˜€•_ĞÇãív2Y*E Ìçê5pèÛñd»”ŒšÔò ºgÌÕj Ğçãõv8Ù-e"Tæ@µÈûé|±·¸‰MY
E8Ìíj2PêCğÎ;ë¬°‚Şx§EVLÁ
/¸ãÍv*Y Å'ìår4Úh¤ÑD£Œ†4öh¹M3ŠjĞõcøÖ=a.W£F7ÚidÑ£°†İ8¦mELòJ:HìÉr)a4×¨¡AG[ÚD¤Ì„ª€öù7ı)~a—·ÑIc‰15+¨àwß™gÕ`ğ×ûá|·	W¹M?ŠoØÓåb4Öh¡G³JHôÉx©A6O©Á8¯­CÂN.K£ˆ†]5hı~sŸšÔñ`»—Ì‘jòúr<Ún$Ó¤¢†|w±KµˆùY}|÷9W­B_Û‡ät–X‘S¼Â.{£œ†16k©3ßª'ÀåoôÓø¢=FnMŠrÚudØÔ¥`„×Ü¡f•=PîCó:¬ô‚8mW’ARO‚KŞH§‰EYÅ:,ìâ26jiÑ3ãª6 é?ñ/û£ü†>/¶cÉ)1!+§ …GÜÍf*U ÀçïõsøÚ=dîT³€ŠØ÷åytİ¦uEÌõj8ĞícòV:A,Ï¢+Æ`­Âqn[“„’’vY2E*LàÊ7èéqq›´”ˆ™SÕ şgÿ•Ğßãçö5y(İ!fg•PğÃûî<³®
¸şº_ÌÇê-pâ[öD¹:lôÒ8¢mF