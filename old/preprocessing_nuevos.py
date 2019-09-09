
import numpy as np
import os
import sys
import progressbar as pb
from imblearn.over_sampling import SMOTE
'''
	Parametros
'''
home = os.environ['HOME']
X_data_dir = home + "/datos_modelo/z_altura{}_2019-04-29_nuevos.npy"
Y_data_dir = home + "/datos_lluvia/precipitacion_nuevos.npy"
'''
53 Cerro Obero 44 nulos. 750 lluvias
37 la cumbrecita 871 nulos. 1092 lluvias
65 Lab Hidraulica 952 nulos.  586 lluvias
'''
estacion = 53
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
#no_rain = np.where(Y[:]==0)[0][:1982]
#Y = np.delete(Y,no_rain,0)
'''
	Concatena varias alturas
'''
x = np.ndarray(shape=(Y.shape[0],96,144,0))
for h in pb.progressbar(alturas):
	'''
	Carga de datos
	'''
	X = np.delete(np.load(X_data_dir.format(h)),missing,0)
	#X = np.delete(X,no_rain,0)
	'''
	Normalizacion y estandarizacion del input
	'''
	u = np.mean(X)
	s = np.std(X)
	X_scaled = (X - u) / s

	X = np.expand_dims(X_scaled, axis=3)

	x = np.concatenate((x,X), axis = 3)

'''
    Contar cuantos 1 hay en total en la estacion.
'''
lluvias = np.where(Y==1)[0].size
nolluvias = np.where(Y==0)[0].size
missing = np.where(Y==-1)[0].size
print("Cant. de datos lluvia en testing: " + str(lluvias))
print("Cant. de datos no lluvia en testing: " + str(nolluvias))
print("Cant. de datos faltantes en testing: " + str(missing))
total = lluvias+nolluvias+missing
total_real = lluvias+nolluvias
print("Total de datos (con faltantes) en testing: " + str(total))
print("Total de datos (sin faltantes) en testing: " + str(total_real))
print("Porcentaje de datos utiles en testing: " + str(total_real/total))
print("Porcentaje de datos lluvia en testing: " + str(lluvias/(total_real)))
print("Porcentaje de datos no lluvia en testing: " + str(nolluvias/(total_real)))

# Shuffle
idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
X = x[idxs]
Y = Y[idxs]

np.save(home + "/datos_modelo/X_Test.npy", X)
np.save(home + "/datos_lluvia/Y_Test.npy", Y)
