import pickle
import scipy.io

FPATH = 'D:/Features/'
name = 'sample'  # probe without 1
p = open(FPATH+'%s.pkl'%name,'rb')
Q = pickle.load(p)
p.close()
dict={}
dict['Q'] = Q
scipy.io.savemat(FPATH+name+".mat",dict)

# name = 'O_objs'  # probe without 1
# p = open('%s.pkl'%name,'rb')
# Ypred, Y = pickle.load(p)
# dict={}
# dict['Ypred'] = Ypred
# dict['Y'] = Y
# scipy.io.savemat(name+".mat",dict)

# name = 'test_objs'  # probe without 1
# p = open('%s.pkl'%name,'rb')
# Ypred, Y, Cpred, C = pickle.load(p)
# dict={}
# dict['Ypred'] = Ypred
# dict['Y'] = Y
# dict['Cpred'] = Cpred
# dict['C'] = C
# scipy.io.savemat(name+".mat",dict)

# name = 'test2_rni_objs'
# p = open('%s.pkl'%name,'rb')
# rni = pickle.load(p)
# dict={}
# dict['rni'] = rni
# scipy.io.savemat(name+".mat",dict)