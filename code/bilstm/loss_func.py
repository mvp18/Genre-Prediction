from keras.metrics import binary_crossentropy
import keras.backend as K

def supply_binlossfunc3(wout): # assuming weight size is (1, output_size, 2)
	def w_binary_crossentropy(target, output):
			output = K.sigmoid(output)
			ce = K.binary_crossentropy(target, output) # (None, output_size)
			#wout = K.placeholder([16, 1]) * weights # (None, output_size, 2)
			batch_size = tf.shape(ce)[0]

			weight = target * wout[:batch_size, :, 0] + (1 - target) * wout[:batch_size, :, 1]  # (None, output_size)
			loss = weight * ce # (None, output_size)
			return K.mean(loss, axis=None)	# (None, 1)
	return w_binary_crossentropy
