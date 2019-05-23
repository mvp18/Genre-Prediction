import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import nltk
import csv
from tqdm import tqdm
import json
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MultiLabelBinarizer


class wordRemover():
    def __init__(self, word):
        self.word = word
        
    def removeWord(self, listOfWords):
        if self.word in listOfWords:
            index = listOfWords.index(self.word)
            del listOfWords[index]
        return listOfWords

def clean_text(text):
    # remove a string like {{plot}}
    text = re.sub("\s*{{\w*}}\s*", "", text)
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text)
    return text

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

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

indices_for_slice = []

running_sent_counter = 0

sentence_list = []

for x in movies_new[['movie_id', 'clean_plot']].values.tolist():
        
    sentences = sent_tokenize(x[1])

    indices_for_slice.append((running_sent_counter, running_sent_counter+len(sentences)))

    running_sent_counter += len(sentences)

    for sentence in sentences:

        sentence_list.append(sentence)

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    message_embeddings = session.run(embed(sentence_list))

total = len(movies_new[['movie_id', 'clean_plot']].values.tolist())

processed = 0

embedding_dict={}
    
for x in movies_new[['movie_id', 'clean_plot']].values.tolist():
        
    embedding_dict[x[0]] = message_embeddings[indices_for_slice[processed][0]:indices_for_slice[processed][1]]

    processed += 1

    if processed % 10000 == 0:
        
        print(processed, "/", total)

print('DICTIONARY LENGTH: {}'.format(len(embedding_dict)))

# Save to disk
print('\nSaving entire dictionary to disk...')
np.save("~/Downloads/use_embeddings.npy", embedding_dict)
print('Finished!')


