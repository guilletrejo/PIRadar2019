from keras.models import Sequential
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

'''
    Parametros
'''
muestras_train = 3260
muestras_test = 652
shape = (96,144,1) # grilla de 96x144 con 1 canal (z). si agregamos otras variables de entrada, sera agregar canales?
X_data_dir = "/home/lac/datos_modelo/z_altura15_2017-11-01_nonan.npy"
Y_data_dir = "/home/lac/datos_lluvia/precipitacion.npy"
model_dir = "/home/lac/PIRadar2019/modeloVGG.h5"
cant_epocas = 2
tam_batch = 326 # intentar que sea multiplo de la cantidad de muestras
'''
    Carga de datos; .Las demas alturas seran un apend?
'''
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)

print("Cant. de nulos en Y: "+str(np.isnan(Y).sum()))
print("Cant. de nulos en X: "+str(np.isnan(X).sum()))

'''
    Split y preprocesamiento del dataset
'''
y_train = Y[:muestras_train]
y_test = Y[muestras_train:muestras_train+muestras_test]


x_train = X[:muestras_train]
x_test = X[muestras_train:muestras_train+muestras_test]

'''
    Definicion del modelo
'''
def get_vgg16():
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(64, (3, 3), input_shape=shape, activation='relu', padding='same'))
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
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(122, activation='sigmoid'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=sgd, metrics=['mse','acc'])
    print(model.summary())

    return model

'''
    Creacion del modelo y entrenamiento
'''
model = get_vgg16()

model.fit(x_train, y_train, batch_size=tam_batch, epochs=cant_epocas, verbose=1, validation_data=(x_test, y_test))

model.save(model_dir)
