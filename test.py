import numpy as np
import pandas as pd 
import requests
import calendar
import urllib as url
import datetime

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
    Una vez se tiene una matriz con los nombres de las estaciones, se puede
    hacer un GET a los datos de lluvia leyendo esa matriz
'''

headers = {
    'accept': 'application/json',
    'X-CSRFToken': 'pYUNHfs6swbibmPTuYEMctRKf3Bvf4Fi9SGbzd3qWIUIkvyd2op8n0KMOKr7wNjH',
}

nombre = estaciones_info['nombre'][0]
anio = 2018
data_rain = pd.DataFrame()

for mes in range(1,13):
    dia_fin = calendar.monthrange(anio,mes)[1]
    fin_date = '{:4}-{:02}-{:02}'.format(anio, mes, dia_fin)
    ini_date = '{:4}-{:02}-01'.format(anio, mes)
    get_body = 'ohmc.psi.unc.edu.ar/bdhm/datos/api/mediciones/{}/{}/00:00/{}/00:00/'.format(nombre,ini_date,hour,fin_date,hour)
    get_body = "https://" + url.quote(get_body)
    response = requests.get(get_body, headers=headers)
    data = pd.read_json(response.text)
    data_rain.append(get_prec_p_hour(data))

def get_prec_p_hour(data):
    '''
        Lee el DataFrame de las estaciones meteorologicas, selecciona la columna de Precipitaciones [mm]
        y obtiene un acumulado por hora del total. Las horas se cuentan de manera continua desde el primer dato.
        Return values: DataFrame precipitations_per_hour ; -1 
    '''
    if (not('Precipitacion [mm]' in data.columns)):
        print ("No existe la columna")
    else:
        data_columns = data[['Precipitacion [mm]']]
        if (data_columns.empty or data_columns.dropna().empty):
            print("VACIO")
        else:
            values_horas=np.ndarray(shape=data_columns.size)
            index = -1
            for i in range (0,data_columns.size):
                if(i%6==0):
                    index+=1
                values_horas[i]=int(index)
            values_horas
            data_columns.insert(0, 'Horas',values_horas)

            return precipitations_per_hour = data_columns.groupby(['Horas']).sum()