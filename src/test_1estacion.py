from keras.models import Sequential
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import os
import sys

'''
    Parametros
'''
balance_ratio = float(sys.argv[1])
home = os.environ['HOME']
shape = (96,144,3) # grilla de 96x144 con 3 canales
X_data_dir = home + "/datos_modelo/X_" + str(balance_ratio) + "Smote.npy"
Y_data_dir = home + "/datos_lluvia/Y_" + str(balance_ratio) + "Smote.npy"
model_dir = home + "/modelos/CerroObero/modeloVgg" + str(balance_ratio) + "Smote.h5"
'''
    Carga de datos
'''
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print("------TESTEANDO CON RATIO: " + str(balance_ratio) + "---------------")
print("TOTAL MUESTRAS: " + str(X.shape[0]))
print(X.shape)
print(Y.shape)

'''
    Carga del modelo y testing
'''
model = load_model(model_dir)

# Loss y Binary Accuracy
muestras_test = int(X.shape[0]*0.10)
X_test = X[-muestras_test:]
Y_test = Y[-muestras_test:]
evaluate = model.evaluate(X_test, Y_test, batch_size = 50)
print(evaluate)

# Score de Clase Positiva
P = model.predict(X)
P[P>=0.5] = 1
P[P<0.5] = 0
score_total = np.count_nonzero(P==Y)/float(P.size)
print("Score total: {}".format(score_total))
score_ones = np.count_nonzero(P[Y==1])/float(P[Y==1].size)
print("Score de horas de lluvia: {}".format(score_ones))