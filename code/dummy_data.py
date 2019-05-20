import pandas as pd
import numpy as np
import torch
import json
import nltk
import re
import csv
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

import InferSent
from InferSent.models import InferSent

class wordRemover():
    def __init__(self, word):
        self.word = word
        
    def removeWord(self, listOfWords):
        if self.word in listOfWords:
            index = listOfWords.index(self.word)
            del listOfWords[index]
        return listOfWords


metadata = pd.read_csv("../data/movie.metadata.tsv", sep = '\t', header = None)
metadata.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

plots = []
with open("../data/plot_summaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab') 
    for row in tqdm(reader):
        plots.append(row)

movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

# change datatype of 'movie_id'
metadata['movie_id'] = metadata['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, metadata[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

genres = [] 

# extract genres
for i in movies['genre']: 
    genres.append(list(json.loads(i).values())) 

# add to 'movies' dataframe  
movies['genre_new'] = genres

genres = movies['genre_new'].values.tolist()

binarizer = MultiLabelBinarizer()

y_data = binarizer.fit_transform(genres)

counts = []
categories = binarizer.classes_

for i in range(categories.shape[0]):
    counts.append((categories[i], np.sum(y_data[:,i])))

df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])

x = df_stats[df_stats['#movies']<200]

genres_to_remove = x['genre'].values.tolist()

for word in genres_to_remove:
    movies['genre_new'] = movies['genre_new'].apply(wordRemover(word).removeWord)

movies_new = movies[~(movies['genre_new'].str.len() == 0)]

print('Updated corpus shape : ', movies_new.shape)

ids = movies_new['movie_id'].values.tolist()

reduced_genres = movies_new['genre_new'].values.tolist()

binarizer_new = MultiLabelBinarizer()

y_data_new = binarizer_new.fit_transform(reduced_genres)

print('Labels array shape : {}'.format(y_data_new.shape))

labels_dict = {}

count = 0
for i in range(len(ids)):
    if i==40:
        break
    labels_dict[ids[i]] = y_data_new[i]

print('LABEL DICTIONARY LENGTH: {}'.format(len(labels_dict)))



def clean_text(text):
    # remove a string like {{plot}}
    text = re.sub("\s*{{\w*}}\s*", "", text)
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text)
    return text

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

print('Creating vocabulary for dataset....')

new_vocabulary = []

for movie_id in movies_new['movie_id']:
    
    plot_sr = movies_new[movies_new['movie_id']==movie_id]['clean_plot']
    
    for str_obj in plot_sr:
        plot=str_obj
    
    sentence_list = sent_tokenize(plot)

    new_vocabulary = new_vocabulary + sentence_list

print('\nVocab creation done!')

V = 2
MODEL_PATH = '/dccstor/cmv/MovieSummaries/All_Data/genre_prediction/pretrained_models/infersent%s.pkl' % V
params_model = {'bsize': 128, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
				'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()

W2V_PATH = '/dccstor/cmv/MovieSummaries/All_Data/genre_prediction/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

model.build_vocab_k_words(K=500000)

print('Updating Vocab....')
model.update_vocab(new_vocabulary, tokenize=True)
print('Vocab updated!')

embedding_dict = {}

count = 0

print('Embedding creation begins...')

for movie_id in movies_new['movie_id']:
    
    plot_sr = movies_new[movies_new['movie_id']==movie_id]['clean_plot']
    
    for str_obj in plot_sr:
        plot=str_obj
    
    sentence_list = sent_tokenize(plot)

    embedding_array = model.encode(sentence_list, tokenize=True)

    embedding_dict[movie_id] = embedding_array

    count += 1

    if count==40:
    	break

print('EMBEDDING DICTIONARY LENGTH: {}'.format(len(embedding_dict)))

# Save to disk
print('\nSaving dummy dictionaries to disk...')
np.save("/dccstor/cmv/MovieSummaries/embeddings/infersent_dummy.npy", embedding_dict)
np.save("/dccstor/cmv/MovieSummaries/embeddings/dummy_labels.npy", labels_dict)
print('Finished!')


