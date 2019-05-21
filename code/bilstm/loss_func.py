from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

def weighted_loss(wout): # assuming weight size is (output_size, 2)
	
	def w_binary_crossentropy(y_true, y_pred):
		
		ce = K.binary_crossentropy(y_true, y_pred) # (None, output_size)

		batch_size = tf.shape(ce)[0]

		class_wt = np.repeat(np.expand_dims(wout, axis=0), batch_size, axis=0)

		weight = y_true*class_wt[:, :, 0] + (1 - y_true)*class_wt[:, :, 1]  # (None, output_size)

		loss = weight*ce # (None, output_size)

		return K.mean(loss, axis=None)	# mean loss value
	
	return w_binary_crossentropy
