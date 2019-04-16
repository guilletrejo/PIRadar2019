
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
hora1 = "06:00:00"
hora2 = "18:00:00"
hora = hora1
cant_alturas=2
base =  datetime.date(2019,1,1)
numdays = 3
date_list = [base + datetime.timedelta(days=x) for x in range(0, numdays)]
# dias=np.ndarray(shape=(numdays),dtype=str)
dias = []
for x in range (0,numdays):
    dias.append(date_list[x].strftime("%Y-%m-%d"))
# print dias

Z = np.zeros(shape=(269,269))
Z = np.expand_dims(Z, axis = 0)

ZF = np.zeros(shape=(cant_alturas,269,269))
ZF = np.expand_dims(ZF, axis = 1)

for dia in dias:
    dataDIR = "../Datos/MatricesX/wrfout_A_d01_{}_{}".format(dia,hora)
    '''
    Abrir el dataset como una matriz XARRAY y mostrar informacion de sus parametros
    '''
    DS = xarray.open_dataset(dataDIR)
    #print DS


    '''
    Mostrar informacion de una variable especifica del dataset 
    (se lee el dataset como NETCDF y la info de la variable se muestra como una matriz xarray).
    '''
    #wrf.getvar(ds, 'z', timeidx=wrf.ALL_TIMES, msl=False) # si no se indica timeidx, selecciona el tiempo 0
    #wrf.getvar(ds, 'PHB', timeidx=wrf.ALL_TIMES)
    #print wrf.getvar(ds, 'z', timeidx=wrf.ALL_TIMES, msl=False, units='km').shape



    '''
    Leer el atributo "Description" de la variable indicada del dataset.
    '''
    #print wrf.getvar(ds, 'PH', timeidx=wrf.ALL_TIMES).description

    '''
    # Seleccionar un subset de datos de una variable especifica, indicando las dimensiones a seleccionar.
    # Dimensiones posibles: Time, bottom_top, south_north, west_east, 
    #                       south_north_stag, west_east_stag, soil_layers_stag, bottom_top_stag
    '''
    for h in range (0,2): #Altura con la que se va a crear la matriz
        # Inicializacion con el tiempo 0
        PH_numpy = DS.PH.sel(Time = 6, bottom_top_stag = h ).values #shape (269, 269)
        PHB_numpy = DS.PHB.sel(Time = 6, bottom_top_stag = h ).values #shape (269, 269)
        z = (PH_numpy + PHB_numpy) / 9.81 # shape (269, 269)
        z_ex = np.expand_dims(z, axis = 0) # shape (1, 269, 269)

        #Completo con el resto de los tiempos
        for t in range(7, 18):
            PH_numpy = DS.PH.sel(Time = t, bottom_top_stag = h ).values #shape (269, 269)
            PHB_numpy = DS.PHB.sel(Time = t, bottom_top_stag = h ).values #shape (269, 269)
            zaux = (PH_numpy + PHB_numpy) / 9.81 # shape (269, 269)
            zaux_ex = np.expand_dims(zaux, axis = 0) # shape (1, 269, 269)
            z_ex = np.concatenate((z_ex, zaux_ex), axis=0) # junta z_ex con zaux_ex, 12hs. shape (12, 269, 269)
        # aca z_ex es (12, 269, 269) para la h de la iteracion
        # Z puede ser shape (269, 269)
        # Z expand dims axis = 0 --> (1, 269, 269)
        # Z concatenate con z_ex cada vez q termine for de tiempos
        z_ex_ex = np.expand_dims(z_ex, axis = 0)
        print "a"
        print z_ex_ex.shape
        
        # ZF[h] = np.concatenate((z_ex,ZF[h]), axis = 0)
        # print ZF[h].shape
    # cuando termine el for de alturas, concatena Z con Z
    print ZF.shape
#np.save('z_altura{}_prueba.npy'.format(h),z)

