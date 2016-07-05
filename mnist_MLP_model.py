from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 10 
nb_epoch = 100 #iteration times

if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(60000, 784).astype('float32')
	x_test = x_test.reshape(10000, 784).astype('float32')
	x_train /= 255.0
	x_test /= 255.0

	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	print 'train samples: ', x_train.shape
	print 'test samples: ', x_test.shape

	print 'building the model ...'

	inputs = Input(shape=(784,) )
	hidden = Dense(512, activation='relu')(inputs)
	#hidden = Dropout(0.2)(hidden)
	outputs = Dense(10, activation='softmax')(hidden)

	model = Model(input=inputs, output=outputs )
	model.compile(loss = 'categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	early_stopping = EarlyStopping(monitor='val_loss', patience=2)

	hist = model.fit(x_train, y_train, batch_size=batch_size, verbose=1, nb_epoch=nb_epoch, validation_split=0.1, callbacks=[early_stopping])
	
	print 'evaluate ...'
	score = model.evaluate(x_test, y_test, verbose=1)
	print '\n'
	print 'loss: ', score[0], '\naccuracy: ', score[1]
	
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']

	nb_epoch = len(loss)
	plt.plot(range(nb_epoch), loss, marker='.', label='loss')
	plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
	plt.legend(loc='best', fontsize=10)
	plt.grid()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
	
