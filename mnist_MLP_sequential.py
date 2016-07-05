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
	
	optimizers = [RMSprop, Adadelta, Adagrad]
	results = {}

	for opt in optimizers:
		model = Sequential()

		model.add(Dense(512, input_shape=(784,)))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))

		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))

		model.add(Dense(10))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer=opt(), metrics=['accuracy'])

		early_stopping = EarlyStopping(monitor='val_loss', patience=2)

		results[opt.__name__] = model.fit( x_train, y_train, batch_size=batch_size, verbose=1, nb_epoch=nb_epoch, validation_split=0.1, callbacks=[early_stopping])
	
		print 'evaluate ...'
		score = model.evaluate(x_test, y_test, verbose=1)
		print '\n'
		print 'loss:', score[0]
		print 'accuracy:', score[1]
	
	for k, result in results.items():
		plt.plot(range(len(result.history['acc'])), result.history['acc'], label=k+'_train')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	for k, result in results.items():
		plt.plot(range(len(result.history['val_acc'])), result.history['val_acc'], label=k+'_val')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.show()	
