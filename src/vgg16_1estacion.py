from keras.models import Sequential
from keras import metrics
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, accuracy_score, zero_one_loss, balanced_accuracy_score, average_precision_score
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
#Sacar los mensajes de debugging de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
'''
    Con este se obtuvo un alto accuracy (mas del 90% para la estacion Cerro Obero oversampleada).
    Para correrlo, asegurarse que X_data_dir corresponde a una X con 3 alturas y con los -1 eliminados de X y de Y, 
    ademas de oversampleada con imb_lear (usar preprocessing_1est.py) 
'''

'''
    Parametros
'''
balance_ratio = 1.0
home = os.environ['HOME']
muestras_train = 0
muestras_val = 0
shape = (68,54,3) # grilla de 96x144 con 3 canales
x_train_dir = home + "/datos_modelo/24horas/umbral0.3/X_Train.npy"
x_val_dir = home + "/datos_modelo/24horas/umbral0.3/X_Val.npy"
y_train_dir = home + "/datos_lluvia/24horas/umbral0.3/Y_Train.npy"
y_val_dir = home + "/datos_lluvia/24horas/umbral0.3/Y_Val.npy"
model_dir = home + "/modelos/CerroObero/24horas/umbral0.3/epoca{epoch:02d}.hdf5"
curve_dir = home + "/modelos/CerroObero/24horas/umbral0.3/graficos/prc{}.png"
cant_epocas = 30
tam_batch = 24 # intentar que sea multiplo de la cantidad de muestras

'''
    Definicion de metricas personalizadas para evaluar en cada epoca y Checkpoints.
'''
def curve(precision, recall, epoch, average_precision):
    step_kwargs = ({'step': 'post'}
        if 'step' in signature(plt.fill_between).parameters
        else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(curve_dir.format(epoch))

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_proba_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]
        precision, recall, thresholds = precision_recall_curve(val_targ,val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        _val_balanced_acc = balanced_accuracy_score(val_targ, val_predict)
        _val_zero_one_loss = zero_one_loss(val_targ, val_predict)
        _val_hmf1acc = 2*(_val_f1*_val_acc)/(_val_f1+_val_acc)
        _val_average_precision = average_precision_score(val_targ, val_proba_predict)
        curve(_val_precision, _val_recall, epoch, _val_average_precision)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("| val_f1: {} | val_precision: {} | val_recall {} | val_acc {} | val_balanced_acc {}".format(_val_f1, _val_precision, _val_recall, _val_acc, _val_balanced_acc))
        print("| harmonic_mean val_f1 val_acc: {} | val_zero_one_loss {} ".format(_val_hmf1acc, _val_zero_one_loss))
        return

metrics = Metrics()
checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', verbose=1, save_best_only=False)
callbacks_list = [checkpoint, metrics]

'''
    Definicion del modelo
'''

def get_vgg16():
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    model.add(BatchNormalization(axis=3, input_shape=shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 4
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    #adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    return model

'''
    Creacion del modelo
'''
model = get_vgg16()

'''
    Carga de datos
'''
x_train = np.load(x_train_dir)
x_val = np.load(x_val_dir)
y_train =  np.expand_dims(np.load(y_train_dir),axis=1)
y_val = np.expand_dims(np.load(y_val_dir),axis=1)

'''
    Entrenamiento
'''
model.fit(x_train[:1000], y_train[:1000], batch_size=tam_batch, epochs=cant_epocas, verbose=1, callbacks=callbacks_list, validation_data=(x_val, y_val))
