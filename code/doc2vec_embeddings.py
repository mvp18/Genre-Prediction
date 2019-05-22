from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import torch
import json
import re
import csv
import pickle
from tqdm import tqdm

def clean_text(text):
    # remove a string like {{plot}}
    text = re.sub("\s*{{\w*}}\s*", "", text)
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text)
    
    text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
    
    text = ' '.join(text.split())
    
    return text

def doc2vec(data_df):
    data = []
    print("Building TaggedDocuments")
    total = len(data_df[['movie_id', 'clean_plot']].values.tolist())
    processed = 0
    for x in data_df[['movie_id', 'clean_plot']].values.tolist():
        label = ["_".join(x[0].split())]
        words = []
        sentences = sent_tokenize(x[1])
        for s in sentences:
            words.extend([x for x in word_tokenize(s)])
        doc = TaggedDocument(words, label)
        data.append(doc)

        processed += 1
        if processed % 10000 == 0:
            print(processed, "/", total)

    print('Done!')

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-5, negative=5, workers=2, epochs=20, min_alpha=0.00025)
    
    print("Building Vocabulary")
    model.build_vocab(data)
    print('Done!')

    print("Training starts")
    
    model.train(documents=data, total_examples=model.corpus_count, epochs=model.epochs)

    print('\nTraining complete')
    
    print('\nBuilding doc2vec vectors')
    # Build doc2vec vectors
    x_data = []
    genres = data_df['genre_new'].values.tolist()
    binarizer = MultiLabelBinarizer()
    y_data = binarizer.fit_transform(genres)
    ids = data_df[['movie_id']].values.tolist()
    for i in range(len(ids)):
        movie_id = ids[i][0]
        label = "_".join(movie_id.split())
        x_data.append(model.docvecs[label])

    return np.array(x_data), y_data

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

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

x_data, y_data = doc2vec(movies_new)

print('X data shape:', x_data.shape)

print('Y data shape:', y_data.shape)

print('Saving data to disk')

with open('/dccstor/cmv/MovieSummaries/embeddings/doc2vec_embeddings.pkl', 'wb') as handle:
    pickle.dump((x_data, y_data), handle)

print('Finished!')