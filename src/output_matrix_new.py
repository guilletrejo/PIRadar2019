'''
Importamos las librerias necesarias para procesar los datos.
'''
import numpy as np
import pandas as pd
import urllib as url
import calendar
import datetime as dt
import progressbar as pb
#np.set_printoptions(threshold=sys.maxsize) # Para que las matrices se impriman completas y no resumidas

''' 
    Parametros
'''

umbral_mm = 0.2
intervalo_minutos = 10
freq = str(intervalo_minutos)+"min"
fecha_inicial = "2019-04-29 00:00"
fecha_final = "2019-07-31 11:50"
nombre_columna_fecha = 'Fecha'
nombre_columna_lluvia = 'Registro de Lluvia [mm]'
precipitation_path = "../../datos_lluvia/"

'''
Lee el archivo Excel de cada anio con las 131 estaciones, carga los nombres en una lista. 
Despues esa lista la usa para recorrer un diccionario de DataFrames (parecido a un array 
pero se puede indexar con una lista) llamado datos2017/8/9. Entonces se va leyendo hoja 
por hoja el excel, y cada hoja se convierte en un DataFrame con 2 columnas (Fecha e Intensidad de Lluvia [mm]),
y cada DataFrame se va guardando en el diccionario datos2017/8/9 (se dropea el anio 2018 para el archivo 2017
asi quedan solo de ese anio, ya que la API de Omixom me hacia tomar minimo 3 meses para que me lo mande al excel, 
entonces yo tome desde 1-11 de 2017 hasta 31-ene de 2018). 
Despues se elimina la columna indices pasando a ser la Fecha el indice, y se completan los valores de fechas faltantes.

Al final, se tienen 3 diccionarios de DataFrames, cada uno con 131 DataFrames por estacion. 
Despues se pasa todo a un solo diccionario con los 131 DataFrames, cuya fecha inicial es
el 1-11-2017 00:00hs y la final es 28-04-2019 12:00hs

### AGREGAR DESPUES SI SE HACE NECESARIO, EL CONTROL AUTOMATICO DE LAS HOJAS CON PROBLEMAS 
### TIPO 'A' (existe la columna fechas pero no tiene nada en lluvia, por lo que aparece 0 cuando deberia aparecer NAN) 
### o TIPO 'B' (el intervalo no es cada 10min)
'''

# Datos 2019

excel = pd.ExcelFile(precipitation_path+"2019_completo.xls")
lista_nombres = excel.sheet_names
datos2019 = {}

idx = pd.date_range(fecha_inicial, fecha_final, freq=freq).strftime("%Y-%m-%d %H:%M:%S")
idx = pd.DatetimeIndex(idx)

for nombre in pb.progressbar(lista_nombres):
    datos2019[nombre] = pd.read_excel(excel,sheet_name=nombre, header=3, parse_dates=[nombre_columna_fecha],dtype={nombre_columna_lluvia: np.float64},dayfirst=True) # lo parsea yyyy-dd-mm
    datos2019[nombre] = datos2019[nombre].set_index([nombre_columna_fecha])
    datos2019[nombre] = datos2019[nombre][~datos2019[nombre].index.duplicated()]
    datos2019[nombre] = datos2019[nombre].reindex(idx)

# Datos totales

datos_total = {}
for nombre in lista_nombres:
    datos_total[nombre] = pd.concat([datos2019[nombre]],sort=False)
    #datos_total[nombre] = pd.concat([datos2017[nombre],datos2018[nombre],datos2019[nombre]],sort=False)   

''' 
    Convierte los dataframe a una lista de matrices, suma el acumulado de lluvia en 1 hora
    y despues convierte todos los valores en binario. Por lo tanto al final se tiene un arreglo 
    de 131 (cant de estaciones) matrices, cada una con 13044 filas (cant. de horas desde 1-11-2017 00:00hs hasta 28-04-2019 12:00hs) 
    y con un valor de 1 si ese dia llovio o 0 si no llovio.
'''

cant_estaciones = int(len(lista_nombres))
cant_horas = int(len(datos_total[lista_nombres[0]]) / (60 / intervalo_minutos))  # Se determina con la cantidad de datos totales dividido por la cantidad de datos por hora
precip_p_estacion = np.ndarray(shape=(cant_horas,cant_estaciones))
no_data_count = 0
# El siguiente bucle recorre la matriz y va sumando el acumulado de 1 hora cada 10 minutos
for estacion in pb.progressbar(lista_nombres):
    temp_data = datos_total[estacion]
    data_columns = temp_data[[nombre_columna_lluvia]]
    if (data_columns.empty or data_columns.dropna().empty):
        print("No hay datos en la estacion: ") + estacion
        no_data_count += 1
        precip_p_estacion[:,lista_nombres.index(estacion)].fill(-1)
    else:
        values_horas = np.ndarray(shape=data_columns.size)
        index = -1
        for i in range(0,data_columns.size):
            if(i % (60 / intervalo_minutos) == 0):
                index += 1
            values_horas[i] = int(index)
        data_columns.insert(0, 'Horas', values_horas)
        precipitations_per_hour = data_columns.groupby(['Horas']).sum(min_count = 1)
        precip_p_estacion[:,lista_nombres.index(estacion)] = precipitations_per_hour.values[:,0]
print("Cantidad de estaciones sin dato: " + str(no_data_count))

# Convierte a 1 si llovio o 0 si no llovio
for estacion in lista_nombres:
    for i in range(cant_horas):
        if (precip_p_estacion[i][lista_nombres.index(estacion)] >= umbral_mm):
            precip_p_estacion[i][lista_nombres.index(estacion)] = 1
        if (precip_p_estacion[i][lista_nombres.index(estacion)] < umbral_mm):
            precip_p_estacion[i][lista_nombres.index(estacion)] = 0
        if (np.isnan(precip_p_estacion[i][lista_nombres.index(estacion)])):
            precip_p_estacion[i][lista_nombres.index(estacion)] = -1

print (precip_p_estacion.shape)

np.save(precipitation_path+'precipitacion_nuevos.npy', precip_p_estacion)
