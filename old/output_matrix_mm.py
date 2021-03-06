'''
Importamos las librerias necesarias para procesar los datos.
'''
import numpy as np
import pandas as pd
import xarray
import netCDF4
#import wrf 
import sys
import urllib as url
import calendar
import datetime as dt
import progressbar as pb
#np.set_printoptions(threshold=sys.maxsize) # Para que las matrices se impriman completas y no resumidas

def get_submatrix(matrix,row,col):
    matrix_aux = np.zeros(shape=(3,3),dtype=float)
    matrix_aux[0,0] = (matrix[row-1,col-1] if row-1>=0 and col-1>=0 else matrix[row,col] )
    matrix_aux[0,1] = (matrix[row-1,col]   if row-1>=0 else matrix[row,col] )
    matrix_aux[0,2] = (matrix[row-1,col+1] if row-1>=0 and col+1<matrix.shape[1] else matrix[row,col] )
    
    matrix_aux[1,0] = (matrix[row,col-1]   if col-1>=0 else matrix[row,col] )
    matrix_aux[1,1] =  matrix[row,col]
    matrix_aux[1,2] = (matrix[row,col+1]   if col+1<matrix.shape[1] else matrix[row,col] )
    
    matrix_aux[2,0] = (matrix[row+1,col-1] if row+1<matrix.shape[0] and col-1>=0 else matrix[row,col] )
    matrix_aux[2,1] = (matrix[row+1,col]   if row+1<matrix.shape[0] else matrix[row,col] )
    matrix_aux[2,2] = (matrix[row+1,col+1] if row+1<matrix.shape[0] and col+1<matrix.shape[1] else matrix[row,col] )
    
    return matrix_aux

''' 
    Parametros
'''

intervalo_minutos = 10
freq = str(intervalo_minutos)+"min"
fecha_inicial = "2017-11-01 00:00"
fecha_final = "2019-04-28 11:50"
nombre_columna_fecha = 'Fecha'
nombre_columna_lluvia = 'Intensidad de Lluvia [mm]'
precip_dir = './datos_lluvia/precipitacion_mm_menos1.npy'
datos_dir = "./datos_lluvia/"
'''
  Leer los nombres y la ubicacion (x,y) de cada estacion y
  se asigna el Nombre como el indice del DataFrame
'''
nombre_ubic = pd.read_csv("/home/lac/datos_lluvia/NombresEstaciones.csv")
nombre_ubic.set_index(['Nombre Estacion'])

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

# Datos 2017

excel = pd.ExcelFile("/home/lac/datos_lluvia/ClimaReporte2017_131.xlsx")
lista_nombres = excel.sheet_names
datos2017 = {}

idx = pd.date_range(fecha_inicial, "2017-12-31 23:50", freq=freq).strftime("%Y-%m-%d %H:%M:%S")
idx = pd.DatetimeIndex(idx)

for nombre in pb.progressbar(lista_nombres):
    datos2017[nombre] = pd.read_excel(excel,sheet_name=nombre, header=3, parse_dates=[nombre_columna_fecha],dtype={nombre_columna_lluvia: np.float64},dayfirst=True) # lo parsea yyyy-dd-mm
    if datos2017[nombre].Fecha.dtype == '<M8[ns]': #OJO CON ESTO, EN OTRA MAQUINA PUEDE SER >M8[ns]
        datos2017[nombre] = datos2017[nombre][datos2017[nombre].Fecha.dt.year != 2018]
    datos2017[nombre] = datos2017[nombre].set_index([nombre_columna_fecha])
    datos2017[nombre] = datos2017[nombre][~datos2017[nombre].index.duplicated()]
    datos2017[nombre] = datos2017[nombre].reindex(idx)

# Datos 2018

excel = pd.ExcelFile("/home/lac/datos_lluvia/ClimaReporte2018_131.xlsx")
lista_nombres = excel.sheet_names
datos2018 = {}

idx = pd.date_range("2018-01-01 00:00", "2018-12-31 23:50", freq=freq).strftime("%Y-%m-%d %H:%M:%S")
idx = pd.DatetimeIndex(idx)

for nombre in pb.progressbar(lista_nombres):
    datos2018[nombre] = pd.read_excel(excel,sheet_name=nombre, header=3, parse_dates=[nombre_columna_fecha],dtype={nombre_columna_lluvia: np.float64},dayfirst=True) # lo parsea yyyy-dd-mm
    datos2018[nombre] = datos2018[nombre].set_index([nombre_columna_fecha])
    datos2018[nombre] = datos2018[nombre][~datos2018[nombre].index.duplicated()]
    datos2018[nombre] = datos2018[nombre].reindex(idx)

# Datos 2019

excel = pd.ExcelFile("/home/lac/datos_lluvia/ClimaReporte2019_131.xlsx")
lista_nombres = excel.sheet_names
datos2019 = {}

idx = pd.date_range("2019-01-01 00:00", fecha_final, freq=freq).strftime("%Y-%m-%d %H:%M:%S")
idx = pd.DatetimeIndex(idx)

for nombre in pb.progressbar(lista_nombres):
    datos2019[nombre] = pd.read_excel(excel,sheet_name=nombre, header=3, parse_dates=[nombre_columna_fecha],dtype={nombre_columna_lluvia: np.float64},dayfirst=True) # lo parsea yyyy-dd-mm
    datos2019[nombre] = datos2019[nombre].set_index([nombre_columna_fecha])
    datos2019[nombre] = datos2019[nombre][~datos2019[nombre].index.duplicated()]
    datos2019[nombre] = datos2019[nombre].reindex(idx)

# Datos totales

datos_total = {}

for nombre in lista_nombres:
    datos_total[nombre] = pd.concat([datos2017[nombre],datos2018[nombre],datos2019[nombre]],sort=False)   

''' 
    Convierte los dataframe a una lista de matrices, suma el acumulado de lluvia en 1 hora
    y despues convierte todos los valores en binario. Por lo tanto al final se tiene un arreglo 
    de 131 (cant de estaciones) matrices, cada una con 13044 filas (cant. de horas desde 1-11-2017 00:00hs hasta 28-04-2019 12:00hs) 
    y con un valor de 1 si ese dia llovio o 0 si no llovio.
'''

cant_estaciones = len(lista_nombres)
cant_horas = len(datos_total[lista_nombres[0]]) / (60 / intervalo_minutos)  # Se determina con la cantidad de datos totales dividido por la cantidad de datos por hora
precip_p_estacion = np.ndarray(shape=(cant_estaciones,int(cant_horas)))
no_data_count = 0
# El siguiente bucle recorre la matriz y va sumando el acumulado de 1 hora cada 10 minutos
for estacion in pb.progressbar(lista_nombres):
    temp_data = datos_total[estacion]
    data_columns = temp_data[['Intensidad de Lluvia [mm]']]
    if (data_columns.empty or data_columns.dropna().empty):
#        print("No hay datos en la estacion: ") + estacion
        no_data_count += 1
        precip_p_estacion[lista_nombres.index(estacion)].fill(-1.0)
    else:
        values_horas = np.ndarray(shape=data_columns.size)
        index = -1
        for i in range(0,data_columns.size):
            if(i % (60 / intervalo_minutos) == 0):
                index += 1
            values_horas[i] = int(index)
        data_columns.insert(0, 'Horas', values_horas)
        precipitations_per_hour = data_columns.groupby(['Horas']).sum(min_count = 1)
        precip_p_estacion[lista_nombres.index(estacion)] = precipitations_per_hour.values[:,0]
print ("Cantidad de estaciones sin dato: " + str(no_data_count))

# Convierte a -1 si no habia datos
for estacion in lista_nombres:
    for i in range(len(precip_p_estacion[0])):
        if (np.isnan(precip_p_estacion[lista_nombres.index(estacion)][i])):
            precip_p_estacion[lista_nombres.index(estacion)][i] = -1.0

'''
    Llenar la matriz Y mapeando las estaciones en su ubicacion correspondiente, cada hora. 
    Como la grilla se achico a 86x135 (estaciones en CBA), se hace el desplazamiento para que los valores (x,y) coincidan

    EN LOS PUNTOS DONDE NO HAY DATOS DE LLUVIA, SE DEJA NaN. Despues ver otras alternativas
    como promediar con las estaciones cercanas, etc...
'''

matrizY = np.zeros([cant_horas,96,144], dtype=np.float64)
matrizY.fill(-1.0)

for hora in pb.progressbar(range(cant_horas)):
    for estacion in lista_nombres:
        index_estacion = lista_nombres.index(estacion)
        x = nombre_ubic.at[index_estacion,'x'] - 65
        y = nombre_ubic.at[index_estacion,'y'] - 69
        matrizY[hora][x][y] = precip_p_estacion[index_estacion][hora]

for hora in pb.progressbar(range(cant_horas)):
    while(np.isnan(matrizY[hora]).any()):
        for cur_pos,cur_val in np.ndenumerate(matrizY[hora]):
            if(np.isnan(cur_val)):
                matrizY[hora][cur_pos] = np.nanmean(get_submatrix(matrizY[hora],cur_pos[0],cur_pos[1]))

np.save('/home/lac/datos_lluvia/precipitacion_mm_av.npy', matrizY)

