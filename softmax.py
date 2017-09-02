import os
import numpy as np
np.random.seed(1234)  # for reproducibility os.getpid()) #

from keras.datasets import mnist
from keras.utils import np_utils

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten

cnn = Sequential()
cnn.add( Conv2D(10, (3,3), padding='same', use_bias=False, data_format='channels_first', input_shape=(1,28,28)) )
cnn.add( BatchNormalization(axis=1) )
cnn.add( Activation('relu') )
cnn.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
cnn.add( Conv2D(10, (3,3), padding='same', use_bias=False, data_format='channels_first') )
cnn.add( BatchNormalization(axis=1) )
cnn.add( Activation('relu') )
cnn.add( Conv2D(10, (3,3), padding='same', use_bias=False, data_format='channels_first') )
cnn.add( BatchNormalization(axis=1) )
cnn.add( Activation('relu') )
cnn.add( Flatten() )
cnn.add( Dense(10, activation='softmax') )
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print (cnn.summary())

cnn.fit(X_train, Y_train, batch_size=64, epochs=5, verbose=1)
score = cnn.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])
