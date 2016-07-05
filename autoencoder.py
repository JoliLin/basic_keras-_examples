import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

def draw_mnist(data, row, col, n):
	size = int(np.sqrt(data.shape[0]))
	plt.subplot(row, col, n)
	plt.imshow(data.reshape(size, size))
	plt.gray()

def draw_figs( figs, show_size ):
	total = 0
	plt.figure(figsize=(20,20))
	for i in xrange(show_size):
		for j in xrange(show_size):
			draw_mnist(figs[total], show_size, show_size, total+1)

			total+=1

	plt.show()

#preprocess
batch_size = 128
input_unit_size = 28*28
nb_epoch = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], input_unit_size)
x_train = x_train.astype('float32')
x_train /= 255

print 'x_train shape: ', x_train.shape
print x_train.shape[0], ' training samples'

#model
inputs = Input(shape=(input_unit_size, ))
x = Dense(144, activation='relu')(inputs)
outputs = Dense(input_unit_size)(x)
model = Model(input=inputs, output=outputs)
model.compile(loss='mse', optimizer='RMSprop')

early_stopping = EarlyStopping(monitor='loss', patience=5)
model.fit(x_train, x_train, batch_size=batch_size, verbose=1, nb_epoch=nb_epoch,callbacks=[early_stopping])
	
#show figures
show_size = 10
#input layer
draw_figs(x_train, show_size)

#hidden layer
get_hidden = K.function([model.layers[0].input], [model.layers[1].output])
hidden_output = get_hidden([x_train[0:show_size**2]])[0]
draw_figs(hidden_output, show_size)

#output layer
get_output = K.function([model.layers[0].input], [model.layers[2].output])
output_output = get_output([x_train[0:show_size**2]])[0]
draw_figs(output_output, show_size)

