import numpy as np
import pandas as pd

Y_data_dir = "../../datos_lluvia/precipitacion.npy"
nombres_dir = "../../nombres_orig.csv"
Y = np.load(Y_data_dir).astype(int)

'''
    Verificar cual es la estacion con menor cantidad de datos faltantes
    y mostrar cuantos datos le faltan
'''
estaciones = np.zeros(shape=(Y.shape[1]))
best=0
cont_best=Y.shape[0]
for actual in range(Y.shape[1]):
    cont_actual=0
    for i in range(Y.shape[0]):
        if(np.equal(Y[i,actual],-1)):
            cont_actual += 1
    if (cont_actual<cont_best):
        best = actual
        cont_best=cont_actual
    estaciones[actual] =  cont_actual/13044 * 100
print("Índice de la estacion con menor cantidad de datos nulos: " + str(best))
print("Cantidad de datos nulos de dicha estación: " + str(cont_best))

best_index = np.where(estaciones<=20.)[0]
nombre_ubic = pd.read_csv(nombres_dir)
print("Estaciones con menos del 20% de datos nulos: \n" + str(nombre_ubic.loc[best_index]))