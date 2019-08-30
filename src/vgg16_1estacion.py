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
    Con este se obtuvo un alto accuracy (mas del 90% para la estacion Cerro Obero oversampleada).
    Para correrlo, asegurarse que X_data_dir corresponde a una X con 3 alturas y con los -1 eliminados de X y de Y, 
    ademas de oversampleada con imb_lear (usar preprocessing_1est.py) 
'''

'''
    Parametros
'''
balance_ratio = float(sys.argv[1])
home = os.environ['HOME']
muestras_train = 0
muestras_test = 0
shape = (96,144,3) # grilla de 96x144 con 3 canales
x_train_dir = home + "/datos_modelo/X_" + str(balance_ratio) + "Train.npy"
x_test_dir = home + "/datos_modelo/X_" + str(balance_ratio) + "Val.npy"
y_train_dir = home + "/datos_lluvia/Y_" + str(balance_ratio) + "Train.npy"
y_test_dir = home + "/datos_lluvia/Y_" + str(balance_ratio) + "Val.npy"
model_dir = home + "/modelos/CerroObero/modeloVgg" + str(balance_ratio) + "TyV.h5"
cant_epocas = 20
tam_batch = 50 # intentar que sea multiplo de la cantidad de muestras

'''
    Definicion del modelo y custom metric
'''

def get_vgg16():
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC layers
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[metrics.binary_accuracy])
    print(model.summary())

    return model

'''
    Creacion del modelo
'''
model = get_vgg16()

'''
    Carga de datos
'''
x_train = np.load(x_train_dir)
x_test = np.load(x_test_dir)
y_train =  np.expand_dims(np.load(y_train_dir),axis=1)
y_test = np.expand_dims(np.load(y_test_dir),axis=1)

'''
    Entrenamiento
'''
model.fit(x_train, y_train, batch_size=tam_batch, epochs=cant_epocas, verbose=1, validation_data=(x_test, y_test))
model.save(model_dir)