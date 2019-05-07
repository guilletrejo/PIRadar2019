import numpy as np
import pandas as pd 
import requests
import calendar


'''
    Se lee el archivo de informacion de las estaciones desde la web.
'''
estaciones_path = "./est_info.txt"
headers = 
{
    'accept': 'application/json',
    'X-CSRFToken': 'CzTcXlGjNXe7ewKzdFGhjSvRfiRKYZDPmtFAPjhDh9XxnFtTL5rDupoTOZHmfIhe',
}
response = requests.get('https://ohmc.psi.unc.edu.ar/bdhm/metadatos/api/estacion/', headers=headers)
file = open(estaciones_path, "w+")
file.write(response.text)
file.close()

'''
    Se lee el archivo json y se convierte a DataFrame de Pandas,
    luego descarta las estaciones que no estan operativas.
'''
estaciones_info = pd.read_json(estaciones_path)
estaciones_info = e
staciones_info[estaciones_info.estado_operativo != False]
estaciones_info['nombre']

'''
    Una vez se tiene una matriz con los nombres de las estaciones, se puede
    hacer un GET a los datos de lluvia leyendo esa matriz
'''
headers = 
{
    'accept': 'application/json',
    'X-CSRFToken': 'pYUNHfs6swbibmPTuYEMctRKf3Bvf4Fi9SGbzd3qWIUIkvyd2op8n0KMOKr7wNjH',
}

nombre = estaciones_info['nombre'][0]
fin_date = '2019-01-31'
ini_date = '2019-01-01'
hour = '00:00'
get_body = 'ohmc.psi.unc.edu.ar/bdhm/datos/api/mediciones/{}/{}/{}/{}/{}/'.format(nombre,fin_date,hour,ini_date,hour)
get_body = url.quote(get_body)
response = requests.get(get_body, headers=headers)

def get_prec_p_hour(data):
    '''
        Lee el DataFrame de las estaciones meteorologicas, selecciona la columna de Precipitaciones [mm]
        y obtiene un acumulado por hora del total. Las horas se cuentan de manera continua desde el primer dato.
        Return values: DataFrame precipitations_per_hour ; -1 
    '''
    data = pd.read_csv("/home/nestormann/Downloads/MAAySP CBA - Laboratorio de Hidraulica__2019-03-01__2019-03-31.csv") 
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