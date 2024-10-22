from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

def weighted_loss(wout): # assuming weight size is (None, output_size, 2)
	
	def w_binary_crossentropy(y_true, y_pred):
		
		ce = K.binary_crossentropy(y_true, y_pred) # (None, output_size)

		batch_size = K.int_shape(ce)[0]

		weight = y_true*wout[:batch_size, :, 0] + (1 - y_true)*wout[:batch_size, :, 1]  # (None, output_size)

		loss = weight*ce # (None, output_size)

		return K.mean(loss, axis=None)	# mean loss value
	
	return w_binary_crossentropy
