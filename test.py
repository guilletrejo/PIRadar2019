import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

param = 'z'
level = '0'
#x = np.load("/datasets/1980-2016/z_1980_2016.npy")
x = np.load("../Datos/{}_altura{}_2017-11-01.npy".format(param,level))
y = 1000*np.expand_dims(np.load("../Datos/precipitacion.npy"), axis=3)

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :]
y = y[idxs, :, :]
# total muestras: 10435
# 80% para trainning: 8348
# resto para testing
y_train = y[:8348, :, :]
y_test = y[8348:, :, :]

# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]


x_train = x[:8348, :, :]
x_test = x[8348:, :, :]

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(86, 135)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)