import numpy as np
import pandas as pd 
import requests

'''
    Se lee el archivo de informacion de las estaciones desde la web.
'''
estaciones_path = "./est_info.txt"
headers = {
    'accept': 'application/json',
    'X-CSRFToken': 'CzTcXlGjNXe7ewKzdFGhjSvRfiRKYZDPmtFAPjhDh9XxnFtTL5rDupoTOZHmfIhe',
}
response = requests.get('https://ohmc.psi.unc.edu.ar/bdhm/metadatos/api/estacion/', headers=headers)
file = open(estaciones_path, "w+")
file.write(response.text)
file.close()

'''
    Leemos el archivo json y lo convertimos a DataFrame de Pandas.
'''
estaciones_info = pd.read_json(estaciones_path)

''' 
    Descartamos aquellas estaciones que no estan operativas.
'''
estaciones_info = estaciones_info[estaciones_info.estado_operativo != False]
estaciones_info['nombre']

'''
    Lee el archivo entero de las estaciones meteorologicas, selecciona la columna de Precipitaciones [mm]
    y obtiene un acumulado por hora del total. Las horas se cuentan de manera continua desde el primer dato.
'''
data = pd.read_csv("/home/nestormann/Downloads/MAAySP CBA - Laboratorio de Hidraulica__2019-03-01__2019-03-31.csv") 
data_columns = data[['Precipitacion [mm]']]

values_horas=np.ndarray(shape=data_columns.size)
index = -1
for i in range (0,data_columns.size):
    if(i%6==0):
        index+=1
    values_horas[i]=int(index)
values_horas
data_columns.insert(0, 'Horas',values_horas)

precipitations_per_hour = data_columns.groupby(['Horas']).sum()