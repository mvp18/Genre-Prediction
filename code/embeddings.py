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
model = model.cuda()
model = nn.DataParallel(model)

W2V_PATH = '/u/soupaul5/All_Data/genre_prediction/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

file = open("../data/plot_summaries.txt")

sentences=[]
for movie in file.readlines():
	movie_id, plot = movie.split('\t')
	for sentence in plot.split('.'):
		if sentence!='\n' and sentence!='':
			sentences.append(sentence)

file.close()

print('SAMPLE SENTENCE:', sentences[2000])

model.build_vocab_k_words(K=100000)
model.update_vocab(sentences, tokenize=True)

print('Vocab updated!')

f = open("../data/plot_summaries.txt")

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
	print(count)
	if count%10000==0:
		print('PLOTS processed:{}'.format(count))

	dict[movie_id] = embeddings

f.close()

print('DICTIONARY LENGTH:', len(dict))
print('LAST ENTRY:', dict[len(dict)-1])

with open("/u/soupaul5/All_Data/genre_prediction/embeddings/MovieSummaries_embeddings.pkl", 'wb') as handle:
	pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('Finished!')
