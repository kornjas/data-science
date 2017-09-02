import os
import numpy as np
np.random.seed(1234)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

mlp = Sequential()
mlp.add( Dense(100, input_dim=784, activation='sigmoid') )
mlp.add( Dense(10, activation='softmax') )
mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print (mlp.summary())

mlp.fit(X_train, Y_train, batch_size=64, epochs=5, verbose=1)
score = mlp.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])
