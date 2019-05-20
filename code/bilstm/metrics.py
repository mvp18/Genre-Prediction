import keras
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

class Metrics(keras.callbacks.Callback):
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
		y_pred = self.model.predict_generator(self.validation_data[0], use_multiprocessing=True, workers=6)
		y_pred_binarized = np.round(y_pred)
		
		aucroc = roc_auc_score(self.validation_data[1], y_pred, average='micro')
		f1 = f1_score(self.validation_data[1], y_pred_binarized, average='micro')
		av_precision = average_precision_score(self.validation_data[1], y_pred_binarized, average='micro')
		
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