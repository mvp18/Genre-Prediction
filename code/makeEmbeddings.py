import requests
import json
import pickle
import re
import time
import numpy as np

def clean_string(sentence):
	return re.sub(r'\s', ' ', sentence)

def get_embedding(sentence):

	r = requests.post('http://tejsentembedding.mybluemix.net/encode/Infersent', data = {'input':sentence})
	if r.status_code == 200:
		return np.array(r.json()['emeddings'])

	elif r.status_code == 500:
		print("ERROR! sentence: ", sentence)
		print(r)
	else:
		print("sentence: ", sentence)
		# print(r.json())
		return None

def read_files_as_setences(line):
	return [s.strip() for s in f.open(filename).read().split('.')]

f = open("../data/plot_summaries.txt")
dict = {}
for movie in f.readlines():
	movie_id, plot = movie.split('\t')
	embeddings = []
	for sentence in plot.split('.'):
		embeddings.append(get_embedding(sentence))
		time.sleep(1)
	dict[movie_id] = embeddings
	print(embeddings)

print('Final embeddings:',embeddings)
with open("../generated_data/movieSummeries_embeddings.pickle", 'wb') as handle:
	pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
