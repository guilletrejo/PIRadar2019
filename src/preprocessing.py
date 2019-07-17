
import numpy as np
import progressbar as pb
from imblearn.over_sampling import SMOTE
'''
	Parametros
'''
X_data_dir = "/home/awf/datos_modelo/z_altura{}_2017-11-01.npy" #3,8,18,4,9,19,5,10,20
Y_data_dir = "/home/awf/datos_lluvia/precipitacion.npy"
estacion = 53 # Cerro Obero
estacion2 = 43 # Las Varas
alturas=[3,8,18]
'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)
missing = np.where(y[:,estacion]==-1)
missing2 = np.where(y[:,estacion2]==-1)
'''
	Eliminacion de -1 y seleccion de una sola estacion
'''
y1 = np.delete(y,missing,0)
y2 = np.delete(y,missing2,0)
Y = y1[:,estacion]
Y2 = y2[:,estacion2]
#y = np.expand_dims(y,axis=1) ## Ver si hace falta hacer esta expansion

'''
	Concatena varias alturas
'''
x = np.ndarray(shape=(13000,96,144,1))
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

# Flatten
x = np.reshape(x,(x.shape[0],41472))
sm = SMOTE(sampling_strategy='minority', random_state=7)
X_us, Y_us = sm.fit_sample(x,Y)
X = np.reshape(X_us,(X_us.shape[0],96,144,3))
Y = Y_us
'''
'''
	Quedarse con solo 2000 datos para tener 40% de lluvias
'''
indices_lluvias = np.where(y==1)[0]
indices_nolluvias = np.where(y==0)[0][0:1250]
Y = np.concatenate((y[indices_lluvias], y[indices_nolluvias]))
Y2 = np.concatenate((y2[indices_lluvias], y2[indices_lluvias]))
X = np.concatenate((x[indices_lluvias], x[indices_nolluvias]))


idxs = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

'''
    Contar cuantos 1 hay en total en todas las estaciones.
'''
lluvias = np.where(Y==1)[0].size
nolluvias = np.where(Y==0)[0].size
missing = np.where(Y==-1)[0].size
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


lluvias = np.where(Y2==1)[0].size
nolluvias = np.where(Y2==0)[0].size
missing = np.where(Y2==-1)[0].size
print("2 Cant. de datos lluvia: " + str(lluvias))
print("2 Cant. de datos no lluvia: " + str(nolluvias))
print("2 Cant. de datos faltantes: " + str(missing))
total = lluvias+nolluvias+missing
total_real = lluvias+nolluvias
print("2 Total de datos (con faltantes): " + str(total))
print("2 Total de datos (sin faltantes): " + str(total_real))
print("2 Porcentaje de datos utiles: " + str(total_real/total))
print("2 Porcentaje de datos lluvia: " + str(lluvias/(total_real)))
print("2 Porcentaje de datos no lluvia: " + str(nolluvias/(total_real)))


X = X[idxs, :, :, :]
Y1 = np.expand_dims(Y[idxs],axis=1)
Y2 = np.expand_dims(Y2[idxs],axis=1)
Y = np.concatenate((Y1,Y2),axis=1)

np.save("/home/awf/datos_modelo/X_6alt_iter_scaled_smote.npy", X)
np.save("/home/awf/datos_lluvia/Y2est_6alt_iter_scaled_smote.npy", Y)
