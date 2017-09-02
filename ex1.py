
import os
import numpy as np
np.random.seed(1234)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense

# Load CSV
import numpy
filename = 'pima-indians-diabetes.data'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)

X = data[:,0:8]
y = data[:,8:9].reshape(data.shape[0])

mean = [ 3.8, 120.9, 69.1, 20.5, 79.8, 32.0, 0.5, 33.2 ]
sd   = [ 3.4, 32.0, 19.4, 16.0, 115.2, 7.9, 0.3, 11.8 ]
X = (X-mean)/sd

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

mlp = Sequential()
mlp.add( Dense(100, input_dim=8, activation='sigmoid') )
mlp.add( Dense(1, activation='sigmoid') )
mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print (mlp.summary())

mlp.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1)
sc = mlp.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', sc[1])
