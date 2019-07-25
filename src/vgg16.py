from keras import backend as K
from keras.models import Sequential
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import numpy as np

'''
    Parametros
'''
muestras_train = 10400
muestras_test = 2600
shape = (96,144,3) # grilla de 96x144 con 3 canales
X_data_dir = "/home/lac/datos_modelo/X_os_all.npy"
Y_data_dir = "/home/lac/datos_lluvia/Y_os_all.npy"
model_dir = "/home/lac/modelos/modeloVgg122est3alt_SMOTE.h5"
mask_dir = "/home/lac/datos_lluvia/mask_precipitacion.npy"
cant_epocas = 20
tam_batch = 50 # intentar que sea multiplo de la cantidad de muestras
'''
    Carga de datos
'''
X = np.load(X_data_dir)[:13000]
Y = np.load(Y_data_dir)[:13000]
print(X.shape)
print(Y.shape)
y_train = Y[:muestras_train]
y_test = Y[muestras_train:muestras_train+muestras_test]
x_train = X[:muestras_train]
x_test = X[muestras_train:muestras_train+muestras_test]

'''
    Definicion del modelo y custom loss function
'''

def custom_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)),
                                        tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))), axis=-1)


def custom_binary_accuracy(y_true, y_pred):
    t0 = tf.equal(y_true, 0)
    t1 = tf.equal(y_true, 1)
    p0 = tf.equal(tf.round(y_pred), 0)
    p1 = tf.equal(tf.round(y_pred), 1)
    everything = tf.reduce_sum(tf.cast(t0, tf.int32)) + tf.reduce_sum(tf.cast(t1, tf.int32))
    positives = tf.reduce_sum(tf.cast(tf.logical_and(t0, p0), tf.int32)) + tf.reduce_sum(tf.cast(tf.logical_and(p1, t1), tf.int32))
    return positives / everything


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32)) * tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)), 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)), 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32)) * tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32)), 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

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
    model.add(Dense(122, activation='sigmoid'))

    #adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=custom_binary_crossentropy, optimizer=sgd, metrics=[custom_binary_accuracy,precision])
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
non_valid = np.count_nonzero(Y==-1)
score_total = np.count_nonzero(P==Y)/float(P.size-non_valid)
print("Score total: {}".format(score_total))
score_ones = np.count_nonzero(P[Y==1])/float(P[Y==1].size)
print("Score de horas de lluvia: {}".format(score_ones))

model.save(model_dir)


