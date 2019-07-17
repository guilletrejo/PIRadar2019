from keras.models import Sequential
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

'''
    Con este se obtuvo un alto accuracy (mas del 90% para la estacion Cerro Obero oversampleada).
    Para correrlo, asegurarse que X_data_dir corresponde a una X con 3 alturas y con los -1 eliminados de X y de Y, 
    ademas de oversampleada con imb_lear (usar preprocessing_1est.py) 
'''

'''
    Parametros
'''
muestras_train = 19600
muestras_test = 4900
shape = (96,144,3) # grilla de 96x144 con 3 canales
X_data_dir = "/home/lac/datos_modelo/X_3alt_iter_scaled_smote.npy"
Y_data_dir = "/home/lac/datos_lluvia/Y_3alt_iter_scaled_smote.npy"
model_dir = "/home/lac/modelos/modeloVgg3AltSMOTE.h5"
cant_epocas = 20
tam_batch = 56 # intentar que sea multiplo de la cantidad de muestras
'''
    Carga de datos
'''
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print(X.shape)
print(Y.shape)
y_train = Y[:muestras_train]
y_test = Y[muestras_train:muestras_train+muestras_test]
x_train = X[:muestras_train]
x_test = X[muestras_train:muestras_train+muestras_test]

'''
    Definicion del modelo y custom metric
'''

def get_vgg16():
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    #model.add(BatchNormalization(axis=3, input_shape=shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 4
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC layers
    model.add(Flatten())
    #model.add(BatchNormalization(axis=3))
    model.add(Dense(4096, activation='relu'))
    #model.add(BatchNormalization(axis=3))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[metrics.binary_accuracy])
#    print(model.summary())

    return model

'''
    Creacion del modelo y entrenamiento
'''
model = get_vgg16()

model.fit(x_train, y_train, batch_size=tam_batch, epochs=cant_epocas, verbose=1, validation_data=(x_test, y_test))

P = model.predict(X)
P[P>=0.5] = 1
P[P<0.5] = 0
score_total = np.count_nonzero(P==Y)/float(P.size)
print("Score total: {}".format(score_total))
score_ones = np.count_nonzero(P[Y==1])/float(P[Y==1].size)
print("Score de horas de lluvia: {}".format(score_ones))

model.save(model_dir)
