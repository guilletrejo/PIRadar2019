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
time_init = 6                    # Se ignoran las primeras 6 horas de cada archivo.
times = 12                       # Se toman 12 horas de cada archivo. (SON 2 ARCHIVOS POR DIA)
horas = ["06:00:00","18:00:00"]
index_horas = 1                  # Se empieza desde las 18hs del primer dia, para coordinar con las iteraciones siguientes
alturas = range(35)         
base = datetime.date(2019,1,1)  # Primer dia del dataset (SIEMPRE LOS DATOS VAN A INICIAR DESDE EL DIA SIGUIENTE A ESTE DIA INICIAL, DEBIDO
                                 #                         A QUE SE EMPIEZA DESDE LAS 18hs. Y SE IGNORAN LAS PRIMERAS 6hs.)
numdays = 544                      # Cant. total de dias del dataset
date_list = [base + datetime.timedelta(days=x) for x in range(1, numdays)]  # Genera una lista de fechas con intervalo de un dia

# Se arma otro arreglo para obtener la lista de fechas en el formato del nombre del archivo del modelo (aaaa-mm-dd)
dias = []
for x in range (0,numdays-1):
    dias.append(date_list[x].strftime("%Y-%m-%d"))


zh0= np.load('./matrix/z_altura{}_{}.npy'.format(alturas[0],dias[0]))
zh1= np.load('./matrix/z_altura{}_{}.npy'.format(alturas[1],dias[0]))
print zh0.shape
print zh1.shape
