import keras
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from keras.losses import binary_crossentropy

class Metrics(keras.callbacks.Callback):

	def __init__(self, val_generator, batch_size, num_classes):

		self.val_generator = val_generator
		self.batch_size = batch_size
		self.num_classes = num_classes

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.f1 = []
		self.average_precision = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		num_batches = len(self.val_generator)
        
        total = num_batches * self.batch_size
        
        val_pred = np.zeros((total, self.num_classes))
        
        val_true = np.zeros((total, self.num_classes))
        
        for batch in range(num_batches):
            
            xVal, yVal = next(self.val_generator)
            
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size] = self.model.predict(xVal)
            
            val_true[batch * self.batch_size : (batch+1) * self.batch_size] = yVal
            
		val_pred_binarized = np.round(val_pred)
		
		aucroc = roc_auc_score(val_true, val_pred_binarized, average='micro')
		f1 = f1_score(val_true, val_pred_binarized, average='micro')
		av_precision = average_precision_score(val_true, val_pred_binarized, average='micro')
		
		self.aucs.append(aucroc)
		self.f1.append(f1)
		self.average_precision.append(av_precision)
		self.losses.append(logs.get('loss'))
		
		print('\naverage precision : {}, f1 score : {}, aucroc score : {}'.format(av_precision, f1, aucroc))
		
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return