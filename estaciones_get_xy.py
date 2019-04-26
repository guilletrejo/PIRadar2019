'''
    Se obtienen las latitudes y longitudes de cada una de las estaciones meteorologicas
    y se convierten a coordenadas (x,y) del modelo WRF, para poder saber a que puntos
    corresponden y poder hacer corresponder la matriz X con la matriz Y.
    Se abre un archivo WRF de un dia y hora arbitrarios (se supone que no cambia nunca)
    y se utiliza la funcion de wrf ll_to_xy.
    Se genera un arreglo (estaciones_xy) de tuplas (x,y).
'''

import numpy as np
import pandas as pd
import xarray
import netCDF4
import wrf 
import sys
import requests
#np.set_printoptions(threshold=sys.maxsize) # Para que las matrices se impriman completas y no resumidas

'''
    Parametros
'''
hora1 = "06:00:00"
hora2 = "18:00:00"
dia = "2019-03-25"
hora = hora1
dataDIR = "../Datos/MatricesX/wrfout_A_d01_{}_{}".format(dia,hora)
estaciones_path = "./est_info.txt" # directorio del archivo con la informacion de las estaciones (obtenido de la API)

'''
    Abrir el dataset como una matriz NETCDF (tratar de usar esta, y no la xarray debido a que tiene 
    en cuenta el formato definido de wrf)
'''
ds = netCDF4.Dataset(dataDIR)

''' 
    NO NECESARIO, SOLO SE HIZO PARA VER LA COBERTURA DEL CUADRADO
    Obtenemos la longitud y la latitud de los extremos del area geografica
        sd = extremo superior derecho
        ii = extremo inferior izquierdo
        ...etc
 '''
(lat_sd, lon_sd) = wrf.xy_to_ll(ds, 0, 0)
(lat_ii, lon_ii) = wrf.xy_to_ll(ds, 269, 269)
(lat_si, lon_si) = wrf.xy_to_ll(ds, 0, 269)
(lat_id, lon_id) = wrf.xy_to_ll(ds, 269, 0)

'''
    Se lee el archivo de informacion de las estaciones desde la web 
    y se convierte a DataFrame de Pandas. Luego se descartan
    las estaciones que no estan operativas.
'''
headers = {
    'accept': 'application/json',
    'X-CSRFToken': 'CzTcXlGjNXe7ewKzdFGhjSvRfiRKYZDPmtFAPjhDh9XxnFtTL5rDupoTOZHmfIhe',
}
response = requests.get('https://ohmc.psi.unc.edu.ar/bdhm/metadatos/api/estacion/', headers=headers)
estaciones_info = pd.read_json(response.text)
estaciones_info = estaciones_info[estaciones_info.estado_operativo != False]

'''
    Genero un arreglo con los (x,y) de todas las estaciones
'''
cant_estaciones = len(estaciones_info.index)
estaciones_xy = np.zeros([cant_estaciones,2], dtype=int)
for i, (lat, lon) in enumerate(zip(estaciones_info['latitud'], estaciones_info['longitud'])):
    (xi, yi) = wrf.ll_to_xy(ds, lat, lon)
    estaciones_xy[i] = (xi, yi)

print estaciones_xy.shape
print estaciones_xy