'''Import modules'''
import time
import sys
import numpy as np
import pickle as pickle
# import os
# import tensorflow as tf
# from keras import backend as K

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, BatchNormalization
from keras.applications.vgg16 import VGG16
# from VGG16_rn import VGG16, extract_XY_generator
from VGG16_rn import extract_XY_generator
from data_generator_side_info import load_data_generator,load_data_generator_simple
from data_generator_side_info import load_data_generator_List, load_data_generator_Uniform_List,rnEqui,rnHard,rnTra
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
N_Class = 3
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

        rn = Input(shape=(LOW_DIM,))
        sideInfo = Input(shape=(N_Class,))

        x = Flatten(name='flatten')(model_vgg16.output)
        x = merge([x, sideInfo], mode='concat', concat_axis=-1)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bm1')(x)
        x = Dense(LOW_DIM, activation='linear', name='predictions')(x)
        model = Model(input=[model_vgg16.input, sideInfo], output=x)

        weightedRn = merge([rn, x],mode='mul')
        modelRn = Model(input=[model_vgg16.input, sideInfo, rn], output=weightedRn)

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

        
        
        (gen_training, N_train), (gen_val, N_val) = load_data_generator_Uniform_List(ROOTPATH, trainT[:], valT[:], test_txt, wei,valMode=VALMODE, validation=validationRatio,subsampling=ssRatio)

        history=self.networkRn.fit_generator(gen_training,
                                             samples_per_epoch=N_train,
                                             nb_epoch=nbEpoch,
                                             verbose=1,
                                             callbacks=[checkpointer,early_stopping],
                                             validation_data=gen_val,
               ®—Acèñ—4£®Ü]?Üo›Êr5hÙ—x£ùFM1
k∏–çc⁄V$¡$Ø§Éƒû,ó¢FsçtÙÿ∏•MD L® hˇë”ü‚ˆqyù4ñhëS≥Ç
x˜ùyV]ΩŒwÎôpï–Ù£¯Ü=].Fcçq4€®§ÅDüåó⁄dÛî∫åÛ⁄:$Ï‰≤4ähò—UcÄ÷·7˜©yA∂{…©6)?°/«£ÌF2M*J`»◊Èaqõ±TãÄòü’W‡¡wÔôs’ ÙÁ¯µ}HﬁIgâY0≈+Ï‡≤7 ih—c≥ñ
8Û≠z\˛F?ç/⁄c‰÷4°(á°]GÜM]
FxÕjvPŸÂ>4Ô®≥¡J/à„Ÿv%$ı$∏‰çtöXî≈P¨√¬..c£ñ=3Æjê˛ˇ≤? oË”Òb;ñlë≤r
Zxƒ›l¶R|˛^?áØ›CÊN5®¯Å}_ûG◊çaZWÑ¡\ØÜ›>&o•ƒÚ,∫b÷z!Á∂5I(…!i'ë%S§¬Æ|Éû∑±IKâô9U- ‚ˆ_˘˝=~n_ìá“bvVY?ºÔŒ3Î™0ÄÎﬂß˚≈|¨ﬁ'æeOîÀ–®£¡F/ç#⁄f$’$†‰áÙùxñ]QCΩ{ªúåñ4Û®∫Lˇä?ÿÔÂsÙ⁄8§ÌD≤LäJ»ıix—c∂V	9?≠/¬cÓV3Å*†˜«˘m}^rGöMT @®œ¡kÔê≥” "(ÊauòÒU{Ä‹üÊı1xÎùpñ[—£ºÜ;∂lâ2u*X‡≈wÏŸr%dÙ‘∏†çG⁄Md T®¿ÅoﬂìÁ“5bh÷QaóæO≥ã ®ıAxœùk÷P°«æ-O¢K∆H≠	By];Ülùrq[¥ƒà¨ôBp˚õ¸îæè≥€ $®‰Åtüòó’Q`√óÓs≥ö
¯Ω{Œ\´Ü ù?÷o·˜≤9Jm“yb]Fq∫tåÿö%T‰¿¥Ø»ÉÈ^1´Ω@éO€ã‰ò¥ïHê…SÈ1>kØêÉ”ﬁ"'¶eEÃ™;¿ÏØÚ˙~<ﬂÆ'√•n”º¢{Ωévô4ï(ê·S˜Ç9^mí}R^BGéM[äDòÃïj–Û„˙6<È.1#´¶ Ö?‹ÔÊ3ı*8‡ÌwÚYzEÃˆ*9 Ì'ÚezT‹¿¶/≈#ÏÊ25*h‡—w„ôv0ı+¯‡ΩwŒYkÖúÛ÷:!,Á¢5FhÕjsê⁄‰Ú4∫hå—Z#ÑÊµ6È9q-¢tÜXùV|¡/∑£…F)!:g¨’B ŒgÎïpê€”‰¢4ÜhùVsÅ©A3è™¿ÙØ¯É˝^>GØçC⁄N$À§®ÑÅ\üÜ›1fkïêÛ”˙"<Ên5®Úzúﬂ÷'·%w§ŸD•Ñ˙ºˆ9;≠,ÇbVwÅ_µ»˝i~QÉ∑ﬁ	gπM0 kË–±cÀñ(ë!SßÇ^|«û-W¢AFOç⁄x§›D¶LÖ
¯ˆ=y.]#Üf6pÈÒ4ª®åÅZÑ˜‹πf:pÏ€Ú$∫då‘ö îÁ–µc»÷)a!ß±EKå»ö)T· ∑ø…OÈÒ8ª≠LÇJH˜âyY6|È17´©@ÅﬂªÁÃµj–˘c˝>q/õ£‘Ü ù'÷ea◊∞°K«à≠YBEL˚ä<òÓs∞⁄‰¯¥ΩHéI[âô<ï.„≥ˆ
98Ì-rbZVD¡Ø∫Ã˛*?†Ô«ÛÌz2\ÍF0Õ+Í`∞◊À·h∑ëISâ>u/ò„’v Ÿ'Â%t‰ÿ¥•HÑ…\©=?Æo√ìÓ3≤j
P¯√˝n>SØÇﬁ~'ü•Wƒ¡lØí“~"_¶G≈l˙R<¬n.S£Ç}7ûiWëSøÇﬁ{ÁúµV¡9o≠¬r.ZcÑ÷°6©=A.O£ã∆≠5BhŒQkÉêû◊≤!Jgà’Y`≈ÏÒr;ölî“¢s∆Z-‚|∂^	π=M.Jcà÷a5®ÒA{èúõ÷°0á´›@¶O≈Ï¯≤=JnH”âbu1Îµpà€Ÿd•Ñúª÷°:¨˝B>Noãìÿí%Rd¬TÆ@ÉèﬁÁ¥µHà…Yi<ÛÆ:¨˛?æoœìÎ“0¢k∆P≠¬~._£á∆m6RiQ>CØé€æ$è§õƒî¨êÇﬁr'öeT‘¿†Ø«√Ìn2S™B ŒÎüó˚—|£ûΩ1Nkãêòì’R ¬gÓUsÄ⁄‰˜ÙπxçZvDŸ•:Ï¸≤>
o∏”Õb*V`¡Ô±sÀö(î·P∑É…^)°=GÆMCäNÀµhà—YcÖÒ6;©,Å"¶w≈lı8ÚmzR\¬F.M#äf’5`Ë◊Òa{óúëVÅ2™w¿ŸoÂÙÚ8∫mL“J"HÊIu	˘5}(ﬁagóïQP√ÉÓ3∑™	@˘˝;˛løí“{‚\∂F	9:m,“b"VfA∞˚À¸®æOøãœÿ´Â@¥œ»´È@±ÀªËå±ZÑ¯úΩVA;è¨õ¬ÆpÉõﬁß∞ÖK‹»¶)E!Á∫5LË 1hÎëpìõ“¢pÜ[›¶|Ö˜∂9I-	"y&]%d˝æpèõ€‘§†Ñá‹ùfU1 Îøè˚€¸§æèºõŒ´∞Äã≠BséZÑÙú∏ñQ:C¨Œ+æ`èó€—d£îÜù3÷j!Á≥ıJ8»ÌirQC¥Œ´π@ç⁄{‰‹¥¶Ö9\Ì2}*^`«óÌQrCöNÀ∞®ã¡XØÖC‹Œ&+• ÑÁ‹µf’9`ÌÚqz[úƒñ,ë"¶r|Ùﬁ8ß≠EBLŒJ+à‡ôw’`ı¯Ò}{û\óÜ]3Üjˆs˘=4Óh≥ëJàÚzuÿˆ%y$›$¶dÖúñ;—,£¢}=nwìôRp˛[ˇÑø‹èÊı4∏ËçqZ[Ñƒú¨ñ>sØö‘˛ øßœ≈kÏ–≤# f(’!`ÁóıQx√ùnS±æxèù[÷D°á∫LˆJ9Ì9rmRt¬XÆECåŒ+¥‡à∑ŸIe	˘0Ω+Œ`´ó¿ëo”ì‚6riQ4√®ÆCøé€ª‰å¥öî˘PΩŒ~+ü†ó«—mcíVA2O™K¿»ØÈCÒ;ª¨åÇt˜òπUM  ËﬂÒg˚ï|êﬁÁ≤5Jh»—icë±2™xÄ›_ÊGıx˙]|∆^-¢}F^Mä}XﬁEgå’Z ƒÁÏµr⁄yd›¶pÖ‹Ù¶8Ö-\‚F6M)
a8◊≠aBWéA[èÑõ‹î¶Ö3‹Í&0Â+Ù‡∏∑ÕIjI…3È*1 ÎßÖ{‹‹¶&%<‰Ó4≥®äXˇÖ‹ﬂÊ'ı%x‰›t¶XÖ\¸∆>-/¢c∆V-"¶_≈Ï˝r>ZoÑ”‹¢&e=Óp≥õ ®Å{ﬂúß÷a<◊Æ!Cßé[ºƒé,õ¢Üpù÷t°áµ]H∆Im	y2]*F`ÕÍqp€õ‰î¥êàìŸR%d˛TøÄèﬂ€Á‰µtàÿôeU¿Ø˚√¸Æ>Øæœæ+œ†´«¿≠o¬SÓB3éjêÙì¯í=RnBSéBétõòîï