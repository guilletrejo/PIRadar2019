'''
Importamos las librerias necesarias para procesar los datos.
'''
import numpy as np
import pandas as pd
import xarray
import netCDF4
import wrf 
import sys
import urllib as url
import calendar
import datetime as dt
import progressbar
#np.set_printoptions(threshold=sys.maxsize) # Para que las matrices se impriman completas y no resumidas