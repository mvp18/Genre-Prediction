import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, ProgbarLogger
from keras.utils import multi_gpu_model
from keras import backend as K
import h5py
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from config_lstm import *
from model_small import BiLSTM
# from model import BiLSTM
from sklearn.metrics import average_precision_score
from metrics import Metrics
from datagen import DataGenerator
from loss_func import weighted_loss
import argparse

# def average_pr(y_true, y_pred):
# 	yp_b = K.round(y_pred)
# 	return K.constant(value=average_precision_score(K.eval(y_true), K.eval(yp_b), average='micro'), dtype='float')

argparser = argparse.ArgumentParser(description="Bidirectional LSTM for genre prediction.")

argparser.add_argument(
	'-run',
	'--type_of_run',
	help="specify dummy or full run",
	default=0,
	type=int)

argparser.add_argument(
	'-reg',
	'--regularization',
	help="whether or not to apply regularization for lstms",
	default=False,
	type=bool
	)

argparser.add_argument(
	'-reg_wt',
	'--regularization_weight',
	help="regularization_weight_for_recurrent layers",
	default=0.0001,
	type=float
	)


argparser.add_argument(
	'-wt_loss',
	'--weighted_loss',
	help='whether or not to use weighted binary_crossentropy',
	default=False,
	type=bool
	)

args = argparser.parse_args()

# for full run
if args.type_of_run == 0:
	print('Loading full data corpus......')
	embedding_dict = np.load('/dccstor/cmv/MovieSummaries/embeddings/Infersent_embeddings.npy', allow_pickle=True).item()
	labels_dict = np.load('/dccstor/cmv/MovieSummaries/embeddings/labels_dict.npy', allow_pickle=True).item()
	print('\nDone Loading')
# for dummy run
else:
	print('Loading dummy data......')
	embedding_dict = np.load('/dccstor/cmv/MovieSummaries/embeddings/infersent_dummy.npy', allow_pickle=True).item()
	labels_dict = np.load('/dccstor/cmv/MovieSummaries/embeddings/dummy_labels.npy', allow_pickle=True).item()
	print('\nDone Loading')

train_ids, val_ids, train_labels, val_labels = train_test_split(list(labels_dict.keys()), list(labels_dict.values()), test_size=0.2, random_state=42)

train_generator = DataGenerator(mode = 'train', data_dict=embedding_dict, list_IDs=train_ids, labels_dict=labels_dict,
								num_classes=NUM_CLASSES, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

valid_generator = DataGenerator(mode = 'val', data_dict=embedding_dict, list_IDs=val_ids, labels_dict=labels_dict,
								num_classes=NUM_CLASSES, batch_size=VAL_BATCH_SIZE, shuffle=False)
with tf.device('/cpu:0'):
	model = BiLSTM(num_classes=NUM_CLASSES, reg=args.regularization, reg_wt=args.regularization_weight)

# model = BiLSTM(num_classes=NUM_CLASSES, reg=args.regularization, reg_wt=args.regularization_weight)

parallel_model = multi_gpu_model(model, gpus=4)

opt = Adam(lr=LEARNING_RATE)

if args.weighted_loss == False:

	parallel_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=None)
	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=None)

else:

	print('Using weighted loss')

	class_weights = np.load('/dccstor/cmv/MovieSummaries/embeddings/class_balanced_weights.npy', allow_pickle=True)

	class_wts = np.repeat(np.expand_dims(class_weights, axis=0), len(embedding_dict), axis=0)

	loss_function = weighted_loss(K.constant(value=class_wts, dtype='float'))

	parallel_model.compile(loss=loss_function, optimizer=opt, metrics=None)

	# model.compile(loss=loss_function, optimizer=opt, metrics=None)

print(model.summary())

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '_' + timestampTime
suffix = "bilstm_" + timestampLaunch + "_reg_" + str(args.regularization) + "_loss_" + str(args.weighted_loss)

# model_name = "weights.{epoch:02d}-{val_average_pr:.4f}.hdf5"

save_path = "/dccstor/cmv/MovieSummaries/results/" + str(suffix)

if args.type_of_run == 0:
	
	if not os.path.exists(save_path):
	    os.makedirs(save_path)

# checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, model_name), monitor='val_average_pr', verbose=1, 
# 							 save_weights_only=False, save_best_only=True, mode='max')

earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=EARLY_STOPPING, verbose=1, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_LR, verbose=1, min_lr=1e-6, mode='min')

score_histories = Metrics(val_generator=valid_generator, batch_size=VAL_BATCH_SIZE, num_classes=NUM_CLASSES)

model_history = parallel_model.fit_generator(generator=train_generator, validation_data=valid_generator, max_queue_size=10, use_multiprocessing=True, workers=4, verbose=1, 
					callbacks=[reduce_lr, earlyStopping, score_histories], epochs=NUM_EPOCHS, shuffle=True)

# model_history = model.fit_generator(generator=train_generator, validation_data=valid_generator, max_queue_size=10, use_multiprocessing=True, workers=4, verbose=1, 
					# callbacks=[reduce_lr, earlyStopping, score_histories], epochs=NUM_EPOCHS, shuffle=True)

training_loss = model_history.history['loss']
valid_loss = model_history.history['val_loss']

#Storing histories as numpy arrays

np.save(save_path+"train_loss.npy", np.array(training_loss))
np.save(save_path+"valid_loss.npy", np.array(valid_loss))
np.save(save_path+"auc.npy", np.array(score_histories.aucs))
np.save(save_path+"f1.npy", np.array(score_histories.f1))
np.save(save_path+"average_pr.npy", np.array(score_histories.average_precision))
