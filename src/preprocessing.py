import numpy as np
import os
import sys
import progressbar as pb
from imblearn.over_sampling import SMOTE
import random
from sklearn.utils import shuffle

'''
0: datos viejos. 
1: datos test(mayo,junio,julio. 2 lluvias). 
2: datos comp(13% lluvias 3 enero). 
3: datos_test_sep(60 horas --> 07/09/2019 00hs hasta 09/09/2019 12hs).
'''
datos = int(sys.argv[1])
shape = [12592, 1505, 13044] # cant. horas eliminando nulos tanto de entrada como de salida, para cada caso.
umbral = 0.3
'''
	Parametros
'''
home = os.environ['HOME']
if(datos == 0 or datos == 2):
	X_data_dir = home + "/datos_modelo/z_altura{}_2017-11-01_wnan_yx_acotada.npy"
	Y_data_dir = home + "/datos_lluvia/precipitacion_umbral{}.npy".format(umbral)
if(datos == 1):
	X_data_dir = home + "/datos_modelo/z_altura{}_2019-04-29_wnan_yx_acotada.npy"
	Y_data_dir = home + "/datos_lluvia/precipitacion_umbral{}_test.npy".format(umbral)

'''
Cerro Obero 44 nulos y 392 lluvias con umbral 0.3
'''
if(datos == 0 or datos == 2):
	estacion = 53
if(datos == 1):
	estacion = 61

balance_ratio = 1.0
alturas=[3,10,15]
rango_comp = np.arange(10260,10320,1) # para borrar los datos de la comparacion y no se metan en el training|

'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)

if(datos==0 or datos==1):
	missing_output = np.where(y[:,estacion]==-1)

'''
	Eliminacion de -1 de output de Y y seleccion de una sola estacion
'''
if(datos==0):
	y = np.delete(y,rango_comp,0)
if(datos==0 or datos==1):
	y = np.delete(y,missing_output,0)

Y = y[:,estacion]

'''
	Concatena varias alturas
'''
x = np.ndarray(shape=(shape[datos],68,54,0))
for h in pb.progressbar(alturas):
    # Carga de datos y eliminacion de -1 de output de X
	X = np.load(X_data_dir.format(h))
	
	if(datos==0):
		X = np.delete(X,rango_comp,0) # elimina los de comparacion
	if(datos==0 or datos==1):
		X = np.delete(X,missing_output,0) 						  # elimina los nulos output
		missing_input = np.argwhere(np.isnan(X[:,0,0]))[:,0]	  # obtiene nulos input
		X = np.delete(X, missing_input, 0)						  # elimina nulos input
	
	# Concatena las alturas
	X = np.expand_dims(X, axis=3)
	x = np.concatenate((x,X), axis = 3)

# Eliminacion de datos nulos de X de la matriz Y
if(datos == 0 or datos == 1):
	Y = np.delete(Y, missing_input, 0)

if(datos == 0):
	'''
	Mezcla las horas al azar pero en bloques de blocksize tamaño
	'''
	muestras_train = int(x.shape[0]*0.8)
	muestras_test = int(x.shape[0]*0.2)
	# Import data
	data = Y
	data1 = x
	blocksize = 24
	# Create blocks
	blocksY = [data[i:i+blocksize] for i in range(0,len(data),blocksize)]
	blocksX = [data1[i:i+blocksize] for i in range(0,len(data1),blocksize)]
	# shuffle the blocks
	blocksX, blocksY = shuffle(blocksX, blocksY, random_state=0)
	# concatenate the shuffled blocks
	Y = [b for bs in blocksY for b in bs]
	X = [b for bs in blocksX for b in bs]

	y_train = Y[:muestras_train]
	y_test = Y[muestras_train:muestras_train+muestras_test]
	x_train = X[:muestras_train]
	x_test = X[muestras_train:muestras_train+muestras_test]
	# Convertir a array de numpy de nuevo
	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)

	'''
		Oversampling
	'''
	# Calculo del porcentaje para balancear las clases
	data0 = int(np.equal(y_train,0).sum())
	data1 = int( data0 * balance_ratio )
	sample_ratio = {0: data0, 1: data1}
	# Flatten
	x_train = np.reshape(x_train,(x_train.shape[0],54*68*3))
	sm = SMOTE(sampling_strategy=sample_ratio, random_state=7, k_neighbors=int(blocksize/2))
	X_us, Y_us = sm.fit_sample(x_train,y_train)
	# DeFlatten
	x_train = np.reshape(X_us,(X_us.shape[0],68,54,3))
	y_train = Y_us

'''
	Contar cuantos 1 hay en total en la estacion.
'''
lluvias = np.where(Y==1)[0].size
nolluvias = np.where(Y==0)[0].size
print("Horas de lluvia: " + str(lluvias))
print("Horas de no lluvia: " + str(nolluvias))
total_real = lluvias+nolluvias
print("Total de horas: " + str(total_real))
print("Porcentaje de horas de lluvia: " + str(lluvias/(total_real)))
print("Porcentaje de horas de no lluvia: " + str(nolluvias/(total_real)))

if(datos==0):
	# Shuffle de nuevo porq SMOTE los pone a todos juntos a los 1
	data = y_train
	data1 = x_train
	blocksize = 24
	# Create blocks
	blocksY = [data[i:i+blocksize] for i in range(0,len(data),blocksize)]
	blocksX = [data1[i:i+blocksize] for i in range(0,len(data1),blocksize)]
	# shuffle the blocks
	blocksX, blocksY = shuffle(blocksX, blocksY, random_state=0)
	# concatenate the shuffled blocks
	Y = [b for bs in blocksY for b in bs]
	X = [b for bs in blocksX for b in bs]
	x_train = np.asarray(X)
	y_train = np.asarray(Y)
	print("Dimensiones de las matrices x_train, y_train, x_test, y_test:")
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

if(datos == 2):
	x = x[rango_comp]
	Y = Y[rango_comp]
if(datos == 1 or datos == 2):
	print("Dimension matriz x, y")
	print(x.shape, Y.shape)

'''
	Guardado de datos
'''
if(datos == 0):
	np.save(home + "/datos_modelo/24horas/umbral{}/X_Train.npy".format(umbral), x_train)
	np.save(home + "/datos_lluvia/24horas/umbral{}/Y_Train.npy".format(umbral), y_train)
	np.save(home + "/datos_modelo/24horas/umbral{}/X_Val.npy".format(umbral), x_test)
	np.save(home + "/datos_lluvia/24horas/umbral{}/Y_Val.npy".format(umbral), y_test)
if(datos == 1):
	np.save(home + "/datos_modelo/comp_test/umbral{}/X_Test.npy".format(umbral), x)
	np.save(home + "/datos_lluvia/comp_test/umbral{}/Y_Test.npy".format(umbral), Y)
if(datos == 2):
	np.save(home + "/datos_modelo/comp_test/umbral{}/X_Comp.npy".format(umbral), x)
	np.save(home + "/datos_lluvia/comp_test/umbral{}/Y_Comp.npy".format(umbral), Y)