
import numpy as np
import progressbar as pb
from imblearn.over_sampling import SMOTE
'''
	Parametros
'''
Y_data_dir = "/home/lac/datos_lluvia/precipitacion.npy"
estacion = 53

'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int) # convertir a entero . y.shape = n_samplesx122
missing = np.where(y[:,estacion==-1) # Obtener los indices de los nulos
     
'''
	Eliminacion de -1 y seleccion de estaciones
'''
y1 = np.delete(y,missing,0)
y = y1[:,estacion]

'''
	Quedarse con solo 2000 datos para tener 40% de lluvias
'''
indices_lluvias = np.where(y==1)[0]
indices_nolluvias = np.where(y==0)[0][0:1250]
Y = np.concatenate((y[indices_lluvias], y[indices_nolluvias]))


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

X = X[idxs, :, :, :]
Y = Y[idxs]

np.save("/home/awf/datos_modelo/X_6alt_iter_scaled_smote.npy", X)
np.save("/home/awf/datos_lluvia/Y_6alt_iter_scaled_smote.npy", Y)
