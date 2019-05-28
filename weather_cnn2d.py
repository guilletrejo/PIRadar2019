import numpy as np
import os.path
import sys
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Lambda
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.callbacks import CSVLogger

def get_rains():
    arr = np.load("../Datos/precipitacion.npy")
    return arr

def get_era_full(param, level):
    arr = np.load("../Datos/{}_altura{}_2017-11-01.npy".format(param,level))
    return arr.astype(np.float32) / 256.

def train_model():
    model = Sequential()
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu', name='block1_conv1', input_shape=(86,135,2)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu', name='block2_conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax', name='final_fully_connected'))

    # Import data
    params = ["z", "z"]
    levels = [0, 34]

    in1_var = get_era_full(params[0], levels[0])
    in2_var = get_era_full(params[1], levels[1])

    X = np.concatenate((np.expand_dims(in1_var, axis=3), np.expand_dims(in2_var, axis=3)), axis=3)
    Y = get_rains()
    #b = np.zeros((Y.shape[0], 2))
    #b[np.arange(Y.shape[0]), Y] = 1

    adagrad = Adagrad(lr=0.0002)
    model.compile(loss='categorical_crossentropy', optimizer = adagrad, metrics=['accuracy'])

    csv_logger = CSVLogger('test.log')
    model.fit(X, Y, batch_size=100, epochs=2, verbose=1, validation_split=0.2, callbacks=[csv_logger])

    model.save('model_2d_test.h5')

if __name__ == '__main__':
    train_model()
