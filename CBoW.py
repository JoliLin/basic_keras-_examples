import sys
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Embedding, Lambda
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import keras.backend as K

def means(x):
	return K.means(x, axis=1)

if __name__ == '__main__':
	docs = []
	with open(sys.argv[1]) as f:
		docs = f.readlines()

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(docs)
	docs = tokenizer.texts_to_sequences(docs)
	numOfvoc = len(tokenizer.word_index)+1

	print '#doc: ', len(docs)
	print '#voc: ', numOfvoc
	dim = 128	#output dimension
	window = 3	#window size

	#fetch the training texts
	data = []
	target = []
	for doc in docs:
		for index, i in enumerate(doc):
			if index <= window-1 or index >= len(doc)-window:
				continue
			start_i = index-window
			end_i = index+window+1
			inser_data = doc[start_i:end_i]
			inser_data.pop(window)
			data.append(inser_data)
			target.append(i)

	x_train = np.array(data)
	y_train = np_utils.to_categorical(np.array(target), numOfvoc)	

	def means(x):
		return K.mean(x, axis=1)

	#model	
	inputs = Input(shape=(6,), dtype='int32')
	x0 = Embedding(input_dim=numOfvoc, output_dim=dim, init='glorot_uniform', input_length=window*2)(inputs)
	x = Lambda(means, output_shape=(dim,))(x0)
	output = Dense(numOfvoc, init='glorot_uniform', activation='softmax')(x)
	
	model = Model(input=inputs, output=output )

	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	model.fit(x_train, y_train, nb_epoch=100, batch_size=512, verbose=1)

	#output vectors
	vec = model.get_weights()[0]
	f = open('vectors.txt', 'w')
	f.write( ' '.join([str(numOfvoc-1), str(dim), '\n']))

	for word, i in tokenizer.word_index.items():
		f.write(word)
		f.write(' ')
		f.write(' '.join(map(str, list(vec[i,:]))))
		f.write('\n')

	f.close()

	
