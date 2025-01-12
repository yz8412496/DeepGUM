import numpy as np
import scipy.io as sio

ROOTPATH = 'C:/Users/Guest_admin/Downloads/DeepGUM-master/'
C_mat = sio.loadmat(ROOTPATH + 'C_objs.mat')
Y = C_mat['Y']
Ypred = C_mat['Ypred']

idx_Y = np.where(np.squeeze(Ypred) == 0)
idx_M = np.where(np.squeeze(Ypred) == 1)
idx_O = np.where(np.squeeze(Ypred) == 2)
