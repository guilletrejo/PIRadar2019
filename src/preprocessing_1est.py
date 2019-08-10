
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
balance_ratio = 0.0
estacion = 37 # 53 Cerro Obero menos nulos. 37 la cumbrecita 871 nulos y mas lluvias (1092)
alturas=[3,8,18]
'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)
missing = np.where(y[:,estacion]==-1)
'''
	Eliminacion de -1 y seleccion de una sola estacion
'''
y1 = np.delete(y,missing,0)
Y = y1[:,estacion]

'''
	Concatena varias alturas
'''
x = np.ndarray(shape=(Y.shape[0]-len(missing),96,144,0))
for h in pb.progressbar(alturas):
	'''
	Carga de datos
	'''
	X = np.delete(np.load(X_data_dir.format(h)),missing,0)
	
	'''
	Normalizacion y estandarizacion del input
	'''
	u = np.mean(X)
	s = np.std(X)
	X_scaled = (X - u) / s

	X = np.expand_dims(X_scaled, axis=3)

	x = np.concatenate((x,X), axis = 3)

'''
	Oversampling

# Calculo del porcentaje para llevar a un desbalance con mayoria de 1s
data0 = int(np.equal(Y,0).sum())
data1 = int( data0 * balance_ratio )
sample_ratio = {0: data0, 1: data1}
# Flatten
x = np.reshape(x,(x.shape[0],96*144*3))
sm = SMOTE(sampling_strategy=sample_ratio, random_state=7)
X_us, Y_us = sm.fit_sample(x,Y)
X = np.reshape(X_us,(X_us.shape[0],96,144,3))
Y = Y_us
'''
# Shuffle
idxs = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

'''
    Contar cuantos 1 hay en total en la estacion.
'''
lluvias = np.where(Y==1)[0].size
nolluvias = np.where(Y==0)[0].size
missing = np.where(Y==-1)[0].size
print("RATIO = " + str(balance_ratio*100) + "%")
print("Cant. de datos lluvia: " + str(lluvias))
print("Cant. de datos no lluvia: " + str(nolluvias))
print("Cant. de datos faltantes: " + str(missing))
total = lluvias+nolluvias+missing
total_real = lluvias+nolluvias
print("Total de datos (con faltantes): " + str(total))
print("Total de datos (sin faltantes): " + str(total_real))
print("Porcentaje de datos utiles: " + str(total_real/total))
print("Porcentaje de datos lluvia: " + str(lluvias/(total_real)))
print("Porcentaje de datos no lluvia: " + str(nolluvias/(total_real)))


X = x[idxs]
Y = Y[idxs]

np.save(home + "/datos_modelo/2X_" + str(balance_ratio) + "Smote.npy", X)
np.save(home + "/datos_lluvia/2Y_" + str(balance_ratio) + "Smote.npy", Y)
