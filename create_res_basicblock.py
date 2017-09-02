
import os
import numpy as np
np.random.seed(1337)  # for reproducibility or os.getpid() for random

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Input
from keras.layers.merge import Add
from keras.utils import np_utils

def create_res_basicblock(input_shape, k):

    x = Input(shape=(input_shape))

    # residual path
    residual = BatchNormalization(axis=1)(x)
    residual = Activation('relu')(residual)
    residual = Conv2D(k, (3,3), padding='same', use_bias=False, data_format='channels_first')(residual) 

    residual = BatchNormalization(axis=1)(residual)
    residual = Activation('relu')(residual)
    residual = Conv2D(k, (3,3), padding='same', use_bias=False, data_format='channels_first')(residual)

    y = Add()([x, residual]) 
    
    block = Model(inputs=[x], outputs=[y])
    return block

# mnist input =28x28, 10 classes
def create_mnist_cnn():
    model = Sequential()

    ......

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load data and reshape the Tensors
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test  = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test  /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = create_mnist_cnn()

print model.summary()

model.fit(X_train, Y_train, batch_size=64, epochs=5, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
