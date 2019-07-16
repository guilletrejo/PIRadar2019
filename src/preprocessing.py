import numpy as np

'''
	Parametros
'''
X_data_dir = "/home/lac/datos_modelo/z_altura{}_2017-11-01.npy" #3,8,18,4,9,19,5,10,20
Y_data_dir = "/home/lac/datos_lluvia/precipitacion.npy"
estacion = 53

'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)
missing = np.where(y[:,estacion]==-1)
'''
	Eliminacion de -1 y seleccion de una sola estacion
'''
y1 = np.delete(y,missing,0)
y = y1[:,53]
#Y = np.expand_dims(Y,axis=1) ## Ver si hace falta hacer esta expansion

x = np.ndarray(shape=(13044,96,144,1))
for h in range(3,5):
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
	Quedarse con solo 2000 datos para tener 40% de lluvias
'''
indices_lluvias = np.where(y==1)[0]
indices_nolluvias = np.where(y==0)[0][0:1250]
Y = np.concatenate((y[indices_lluvias], y[indices_nolluvias]))
X = np.concatenate((x[indices_lluvias], x[indices_nolluvias]))

idxs = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

#X = X[idxs, :, :, :]
#Y = Y[idxs]

np.save("/home/lac/datos_modelo/X_3alt_scaled_a.npy", X)
np.save("/home/lac/datos_lluvia/Y_int_Cerro_nonulls_balanced_a.npy", Y)
