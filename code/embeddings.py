from random import randint
import numpy as np
import torch
import pickle
import time
import sys

import InferSent
from InferSent.models import InferSent

V = 2
MODEL_PATH = '/u/soupaul5/All_Data/genre_prediction/pretrained_models/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
				'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = '/u/soupaul5/All_Data/genre_prediction/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

f = open("../data/plot_summaries.txt")

sentences=[]
for movie in f.readlines():
	movie_id, plot = movie.split('\t')
	for sentence in plot.split('.'):
		if sentence!='\n' and sentence!='':
			sentences.append(sentence)

model.build_vocab_k_words(K=500000)
model.update_vocab(sentences, tokenize=True)

dict = {}
count=0
for movie in f.readlines():
	movie_id, plot = movie.split('\t')
	count+=1
	embeddings=[]
	for sentence in plot.split('.'):
		if sentence!='\n' and sentence!='':
			try:
				embeddings.append(model.encode(sentence))
			except:
				print('Error with :', movie_id)
				print('SENTENCE: ', sentence)
				print('PLOT: ', plot)
				print('Number of plots processed till now:', count)
				exit()
			time.sleep(1)
	dict[movie_id] = embeddings

f.close()

with open("/u/soupaul5/All_Data/genre_prediction/embeddings/MovieSummaries_embeddings.pkl", 'wb') as handle:
	pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('Finished!')
