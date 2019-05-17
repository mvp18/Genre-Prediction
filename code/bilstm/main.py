import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, ProgbarLogger
from keras import backend as K
import h5py
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from config_lstm import *
from preprocess import preprocess
from model import BiLSTM
from sklearn.metrics import average_precision_score
from metrics import Metrics
from datagen import DataGenerator


def average_pr(y_true, y_pred):
	yp_b = np.round(y_pred)
	return average_precision_score(y_true, yp_b, average='macro')

list_of_ids, labels_tuple_list, labels_array = preprocess(METADATA_PATH, PLOT_SUMMARIES_PATH)

train_ids, val_ids, train_labels, val_labels = train_test_split(list_of_ids, labels_tuple_list, test_size=0.2, stratify=labels_array, random_state=42)

print('Loading data corpus')

embedding_dict = np.load('/dccstor/cmv/MovieSummaries/embeddings/Infersent_embeddings.npy').item()

print('\nDone Loading')

train_generator = DataGenerator(data_dict=embedding_dict, list_IDs=train_ids, labels_dict=dict(train_labels), num_classes=labels_array.shape[1], 
								batch_size=BATCH_SIZE, shuffle=True)

valid_generator = DataGenerator(data_dict=embedding_dict, list_IDs=val_ids, labels_dict=dict(val_labels), num_classes=labels_array.shape[1],
								batch_size=BATCH_SIZE, shuffle=False)

model = BiLSTM(labels_array.shape[1])

opt = Adam(lr=LEARNING_RATE)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[average_pr])

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '_' + timestampTime
# suffix = timestampLaunch + 'bilstm'
suffix = 'bilstm'

save_path = '/dccstor/cmv/MovieSummaries/results/' + str(suffix)

if not os.path.exists(save_path):
    os.makedirs(save_path)

checkpoint = ModelCheckpoint(filepath=save_path+'bilstm_model.h5', monitor='val_average_pr', verbose=1, 
							 save_weights_only=False, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_LR, verbose=1, min_lr=1e-6)

score_histories = Metrics()

model.fit_generator(generator=train_generator, validation_data=valid_generator, use_multiprocessing=True, workers=6, verbose=1, 
					callbacks=[checkpoint, reduce_lr, score_histories], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=True)

#Storing histories as numpy arrays

np.save(save_path+"losses.npy", np.array(score_histories.losses))
np.save(save_path+"auc.npy", np.array(score_histories.aucs))
np.save(save_path+"f1.npy", np.array(score_histories.f1))
np.save(save_path+"average_pr.npy", np.array(score_histories.average_precision))

