import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adadelta
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10

#preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

#variables settings
batch_size = 128
nb_epoch = 10

nb_filters = 32	#numbers of filters
nb_pool = 2	#the window of pooling layer
nb_conv = 3 #the window size of convolution layer

#model
inputs = Input(shape=(1, 28, 28))
x = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', activation='relu')(inputs)
x2 = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu')(x)
x3 = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x2)
x4 = Dropout(0.25)(x3)
x5 = Flatten()(x4)
x6 = Dense(128, activation='relu')(x5)
x7 = Dropout(0.5)(x6)
x8 = Dense(10, activation='softmax')(x7)

model = Model(input=inputs, output=x8)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)
print score

print 'loss:    ', score[0]
print 'accuracy: ', score[1]

