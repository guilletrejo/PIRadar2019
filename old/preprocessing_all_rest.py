import numpy as np
import sys
import os
import progressbar as pb
from imblearn.over_sampling import SMOTE
'''
	Parametros
'''
home = os.environ['HOME']
X_data_dir = home + "/datos_modelo/X_os_all.npy" #3,8,18,4,9,19,5,10,20
Y_data_dir = home + "/datos_lluvia/Y_os_all.npy"
estacion = int(sys.argv[1])
alturas=[3,8,18]

'''
	Carga de datos
'''
Y = np.load(Y_data_dir).astype(int)
X = np.load(X_data_dir)

'''
Oversampling
X -> n_samples, 96, 144, canales
Y -> n_samples, n_estaciones
Y_ind -> n_samples, 1
Yp-> n_estaciones -> representa porcentaje de UNOs en cada estacion
'''
def get_Yp(Y):
    Yp = np.zeros(shape=Y.shape[1])
    for est in range(Y.shape[1]):
        Yp[est] = np.where(Y[:,est]==1)[0].size/Y.shape[0]
    return Yp

# Flatten
Y_ind = Y[:,estacion]                                   # Y_ind es el correspondiente a la estacion particular
X = np.reshape(X,(X.shape[0],96*144*3))          # Hago el flatten de X

nodata = int(np.equal(Y_ind,-1).sum())
data0 = int(np.equal(Y_ind,0).sum())
data1 = int((np.equal(Y_ind,1).sum() + data0)*0.2)
sample_ratio = {-1: nodata, 0: data0, 1: data1}
sm = SMOTE(sampling_strategy=sample_ratio, random_state=7)

X_os, Y_ind_os = sm.fit_sample(X,Y_ind)          # OverSampled X e Y_ind
X = np.reshape(X_os,(X_os.shape[0],96,144,3))    # X vuelve a ser una grilla pero con oversampling
Y_os = np.zeros(shape=(Y_ind_os.shape[0],Y.shape[1]))   # Y_os es una nueva matriz con shape n_oversamples, n_estaciones
Y_os.fill(-1)                                    # Relleno Y_os con -1
Y_os[:Y.shape[0],:] = Y                             # Y_os va a ser igual a Y hasta la cantidad de muestras de Y
Y_os[Y.shape[0]+1:, estacion] = Y_ind_os[Y.shape[0]+1:]        # El resto de Y_os es -1 excepto en la estacion actual
Y = Y_os                                         # Reemplazo Y 
print("Cant. de muestras= "+str(Y.shape[0]))

print(get_Yp(Y)*100)

'''
Shuffle
'''
idxs = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
X = X[idxs]
Y = Y[idxs]
print(Y.shape)
print(X.shape)
np.save(home + "/datos_modelo/X_os_all.npy", X)
np.save(home + "/datos_lluvia/Y_os_all.npy", Y)
