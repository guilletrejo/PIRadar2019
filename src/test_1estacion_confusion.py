from keras.models import Sequential
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix
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
CUTOFF = 0.5
'''
    Carga de datos y modelo
'''
model = load_model(model_dir)
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print("------TESTEANDO CON RATIO: " + str(balance_ratio) + "---------------")
print("TOTAL MUESTRAS: " + str(X.shape[0]))
print(X.shape)
print(Y.shape)

'''
    Testing (y_true = Y ; y_pred = P)
'''
P = model.predict(X)

P[P>=CUTOFF] = 1
P[P<CUTOFF] = 0

TN, FP, FN, TP = confusion_matrix(Y,P).ravel()
accuracy = (TP + TN) / (TP+TN+FP+FN)
precision = (TP) / (TP+FP)
recall = (TP) / (TP+FN)

print("True Positives: {}".format(TP))
print("True Negatives: {}".format(TN))
print("False Positives: {}".format(FP))
print("False Negatives: {}".format(FN))
print("---------------")
print("Accuracy = {}".format(accuracy))
print("Precision = {}".format(precision))
print("Recall = {}".format(recall))