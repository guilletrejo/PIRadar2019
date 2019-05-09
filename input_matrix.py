'''
Importamos las librerias necesarias para procesar los datos.
'''
import numpy as np
import pandas as pd
import xarray
import netCDF4
import wrf 
import sys
import datetime
#np.set_printoptions(threshold=sys.maxsize) # Para que las matrices se impriman completas y no resumidas

'''
Parametros
'''
path_datos = "/home/datos/wrf/wrfout/"  # en YAKU, cambiar esto
time_init = 6                    # Se ignoran las primeras 6 horas de cada archivo.
times = 12                       # Se toman 12 horas de cada archivo. (SON 2 ARCHIVOS POR DIA)
horas = ["06:00:00","18:00:00"]
index_horas = 1                  # Se empieza desde las 18hs del primer dia, para coordinar con las iteraciones siguientes
alturas = range(35)         
base = datetime.date(2017,10,31)  # Primer dia del dataset (SIEMPRE LOS DATOS VAN A INICIAR DESDE EL DIA SIGUIENTE A ESTE DIA INICIAL, DEBIDO
                                 #                         A QUE SE EMPIEZA DESDE LAS 18hs. Y SE IGNORAN LAS PRIMERAS 6hs.)
numdays = 544                      # Cant. total de dias del dataset
date_list = [base + datetime.timedelta(days=x) for x in range(1, numdays)]  # Genera una lista de fechas con intervalo de un dia

# Se arma otro arreglo para obtener la lista de fechas en el formato del nombre del archivo del modelo (aaaa-mm-dd)
dias = []
for x in range (0,numdays-1):
    dias.append(date_list[x].strftime("%Y-%m-%d"))
 
"""
Abrir el dataset como una matriz XARRAY, dando formato al nombre con la listas de fechas y las 2 posibles horas de inicio.
"""
dataDIR = path_datos + "{}/wrfout_A_d01_{}_{}".format(base.strftime("%Y-%m-%d"),horas[index_horas])
DS = xarray.open_dataset(dataDIR)
index_horas ^= 1                 # Se swapea para abrir el proximo archivo del mismo dia con la otra hora (6hs)

"""
Generar los archivos numpy extrayendo la altura geopotencial (z) a partir de 2 variables del modelo: PH y PHB
La formula utilizada es z = (PH + PHB) / 9.81

Cada matriz de datos tiene dimension (86,135) (rectangulo formado por las estaciones de las cuales se tienen datos), y se extraen 12 horas de cada archivo:
    * Se empieza abriendo el archivo del primer dia a las 18hs, se ignoran las primeras 6 (hasta las 23hs inclusive). 
      A partir de ahi se toman 12hs (desde 00am hasta las 11 am). Notar que ya se abarca el dia siguiente (es decir de este
      archivo se tomaron las primeras 12hs del dia sgte.)
    * Despues, se abre el segundo archivo (6hs del dia siguiente) y se ignoran las  primeras 6 (hasta las 11am).
      A partir de ahi se toman 12hs (desde 12pm hasta las 23pm) cubriendo la otra mitad del dia.

    ** Esto se repite con todos los archivos, generando los datos (rectangulo de 86*135 con el dato 'z' en cada punto) para
       todos los dias del dataset en CADA UNA de las 34 alturas posibles del modelo, por lo que las dimensiones de la matriz
       final serian (cant_de_horas,86,135), y se genera UN ARCHIVO (.npy) por cada altura del modelo.
La cantidad de horas obtenidas es (numdays*24horas)-12horas

"""

for h in alturas:                                                          # Alturas en las que se van a generar los archivos (SOLO PARA EL PRIMER DIA)
    # Inicializacion con el tiempo 0 (se toma el primer dato solamente)
    PH_numpy = DS.PH.sel(Time = time_init, bottom_top_stag = h ).values[70:156,73:208]    # Extrae el dato PH en la 1 hora
    PHB_numpy = DS.PHB.sel(Time = time_init, bottom_top_stag = h ).values[70:156,73:208]  # Extrae el dato PHB en la 1 hora
    z = (PH_numpy + PHB_numpy) / 9.81                                      # Obtiene altura geopotencial (matriz 269x269)
    z_ex = np.expand_dims(z, axis = 0)                                     # Agrega una dimension para agregar las otras horas

    # Completa con el resto de los tiempos
    for t in range(time_init+1,time_init+times):
        PH_numpy = DS.PH.sel(Time = t, bottom_top_stag = h ).values[70:156,73:208]        # Extrae el dato PH en la hora t
        PHB_numpy = DS.PHB.sel(Time = t, bottom_top_stag = h ).values[70:156,73:208]      # Extrae el dato PHB en la hora t
        zaux = (PH_numpy + PHB_numpy) / 9.81                               # Obtiene altura geopotencial en la hora t
        zaux_ex = np.expand_dims(zaux, axis = 0)                           # Agrega una dimension asi se puede concatenar con z_ex (primer valor generado)
        z_ex = np.concatenate((z_ex, zaux_ex), axis=0)                     # Concatena las matrices y queda (12x269x269)
    np.save('./datos_modelo/z_altura{}_{}.npy'.format(h,dias[0]),z_ex)           # Guarda 1 archivo por cada altura, con las primeras 12 horas.
    
DS.close() # Se cierra el dataset abierto para ahorrar memoria

for dia in dias:                                                           # Lee el resto de los dias del dataset
    for hora in horas:                                                     # Lee ambas horas de cada dia (6hs y 18hs)
        dataDIR = path_datos + "wrfout_A_d01_{}_{}".format(dia,hora) 
        DS = xarray.open_dataset(dataDIR)                                  # Abre el archivo

        for h in alturas: 
            # Inicializacion con el tiempo 0
            PH_numpy = DS.PH.sel(Time = time_init, bottom_top_stag = h  ).values[70:156,73:208]
            PHB_numpy = DS.PHB.sel(Time = time_init, bottom_top_stag = h ).values[70:156,73:208]
            z = (PH_numpy + PHB_numpy) / 9.81
            z_ex = np.expand_dims(z, axis = 0)

            # Completa con el resto de los tiempos
            for t in range(time_init+1,time_init+times):
                PH_numpy = DS.PH.sel(Time = t, bottom_top_stag = h ).values[70:156,73:208]
                PHB_numpy = DS.PHB.sel(Time = t, bottom_top_stag = h ).values[70:156,73:208]
                zaux = (PH_numpy + PHB_numpy) / 9.81
                zaux_ex = np.expand_dims(zaux, axis = 0)
                z_ex = np.concatenate((z_ex, zaux_ex), axis=0)
            z_file = np.load('./datos_modelo/z_altura{}_{}.npy'.format(h,dias[0]))  # Abre la matriz .npy generada anteriormente
            z_ex = np.concatenate((z_file, z_ex), axis=0)            # Concatena la nueva matriz con la anterior
            np.save('./datos_modelo/z_altura{}_{}.npy'.format(h,dias[0]),z_ex)      # Guarda nuevamente el archivo .npy (se va sobreescribiendo el archivo con los nuevos valores)
        DS.close()