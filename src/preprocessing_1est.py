
import numpy as np
import os
import sys
import progressbar as pb
from imblearn.over_sampling import SMOTE
'''
	Parametros
'''
home = os.environ['HOME']
X_data_dir = home + "/datos_modelo/z_altura{}_2017-11-01.npy" #3,8,18,4,9,19,5,10,20
Y_data_dir = home + "/datos_lluvia/precipitacion.npy"
balance_ratio = 1.0
'''
53 Cerro Obero 44 nulos. 750 lluvias
37 la cumbrecita 871 nulos. 1092 lluvias
65 Lab Hidraulica 952 nulos.  586 lluvias
'''
estacion = 53
alturas=[3,10,19]
'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)
missing_output = np.where(y[:,estacion]==-1)
'''
	Eliminacion de -1 de output de Y y seleccion de una sola estacion
'''
y1 = np.delete(y,missing_output,0)
Y = y1[:,estacion]

'''
	Concatena varias alturas
'''
x = np.ndarray(shape=(Y.shape[0],96,144,0))
for h in pb.progressbar(alturas):
	'''
	Carga de datos y eliminacion de -1 de output de X
	'''
	X = np.delete(np.load(X_data_dir.format(h)),missing_output,0)
	
	'''
	Normalizacion y estandarizacion del input
	'''
	u = np.mean(X)
	s = np.std(X)
	X_scaled = (X - u) / s

	X = np.expand_dims(X_scaled, axis=3)

	x = np.concatenate((x,X), axis = 3)

'''
	Eliminacion nulos de input tanto de X como de Y
'''
#Obtener los indices de los nulos de input
missing_input = np.where(x[:,0,0,0]==np.nan)
x = np.delete(x, missing_input, 0)
Y = np.delete(Y, missing_input, 0)

'''
	Division en entrenamiento y validacion
'''
muestras_train = int(x.shape[0]*0.8)
muestras_test = int(x.shape[0]*0.2)
# Shuffle
idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
X = x[idxs]
Y = Y[idxs]
y_train = Y[:muestras_train]
y_test = Y[muestras_train:muestras_train+muestras_test]
x_train = x[:muestras_train]
x_test = x[muestras_train:muestras_train+muestras_test]

'''
	Oversampling
'''
# Calculo del porcentaje para balancear las clases
data0 = int(np.equal(y_train,0).sum())
data1 = int( data0 * balance_ratio )
sample_ratio = {0: data0, 1: data1}
# Flatten
x_train = np.reshape(x_train,(x_train.shape[0],96*144*3))
sm = SMOTE(sampling_strategy=sample_ratio, random_state=7)
X_us, Y_us = sm.fit_sample(x_train,y_train)
# DeFlatten
x_train = np.reshape(X_us,(X_us.shape[0],96,144,3))
y_train = Y_us

'''
    Contar cuantos 1 hay en total en la estacion.
'''
lluvias = np.where(y_test==1)[0].size
nolluvias = np.where(y_test==0)[0].size
missing = np.where(y_test==-1)[0].size
print("RATIO = " + str(balance_ratio*100) + "%")
print("Cant. de datos lluvia en validacion: " + str(lluvias))
print("Cant. de datos no lluvia en validacion: " + str(nolluvias))
print("Cant. de datos faltantes en validacion: " + str(missing))
total = lluvias+nolluvias+missing
total_real = lluvias+nolluvias
print("Total de datos (con faltantes) en validacion: " + str(total))
print("Total de datos (sin faltantes) en validacion: " + str(total_real))
print("Porcentaje de datos utiles en validacion: " + str(total_real/total))
print("Porcentaje de datos lluvia en validacion: " + str(lluvias/(total_real)))
print("Porcentaje de datos no lluvia en validacion: " + str(nolluvias/(total_real)))

# Shuffle
idxs = np.arange(x_train.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
x_train = x_train[idxs]
y_train = y_train[idxs]

np.save(home + "/datos_modelo/X_" + str(balance_ratio) + "Train.npy", x_train)
np.save(home + "/datos_lluvia/Y_" + str(balance_ratio) + "Train.npy", y_train)
np.save(home + "/datos_modelo/X_" + str(balance_ratio) + "Val.npy", x_test)
np.save(home + "/datos_lluvia/Y_" + str(balance_ratio) + "Val.npy", y_test)