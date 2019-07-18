
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


'''
Oversampling
X -> n_samples, 96, 144, canales
x -> n_samples, 96*144*canales
Y -> n_samples, n_estaciones
Y_ind -> n_samples, 1
Yp-> n_estaciones -> representa porcentaje de UNOs en cada estacion
'''
def get_Yp(Y):
    Yp = np.zeros(shape=Y.shape[1])
    for est in range(Y.shape[1]):
        Yp[est] = np.where(Y[:,est]==1)[0].size/Y.shape[0]
    return Yp

for i in range(122) #Mientras que Yp[i]>=50% para todo i
    # Flatten
    Y_ind = Y[:,i]                                   # Y_ind es el correspondiente a la estacion particular
    x = np.reshape(x,(X.shape[0],96*144*3))          # Hago el flatten de X

    nodata = int(np.equal(Y[:,i],-1).sum())
    data0 = int(np.equal(Y[:,i],0).sum())
    data1 = int((np.equal(Y[:,i],1).sum() + data0 + nodata)*0.2)
    sample_ratio = {-1: nodata, 0: data0, 1: data1}
    sm = SMOTE(sampling_strategy=sample_ratio, random_state=7)

    X_os, Y_ind_os = sm.fit_sample(x,Y_ind)          # OverSampled X e Y_ind
    X = np.reshape(X_os,(X_os.shape[0],96,144,3))    # X vuelve a ser una grilla pero con oversampling
    Y_os = zeros(shape=(Y_ind_os.size,Y.shape[1]))   # Y_os es una nueva matriz con shape n_oversamples, n_estaciones
    Y_os.fill(-1)                                    # Relleno Y_os con -1
    Y_os[0:Y.size,:] = Y                             # Y_os va a ser igual a Y hasta la cantidad de muestras de Y
    Y_os[Y.size+1:Y_ind_os.size , i] = Y_ind_os        # El resto de Y_os es -1 excepto en la estacion actual
    Y = Y_os                                         # Reemplazo Y 



np.save("/home/awf/datos_modelo/X_6alt_iter_scaled_smote.npy", X)
np.save("/home/awf/datos_lluvia/Y_6alt_iter_scaled_smote.npy", Y)
