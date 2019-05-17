import pandas as pd
import numpy as np
import json
import re
import csv
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

class wordRemover():
    def __init__(self, word):
        self.word = word
        
    def removeWord(self, listOfWords):
        if self.word in listOfWords:
            index = listOfWords.index(self.word)
            del listOfWords[index]
        return listOfWords

def preprocess(meta_path, plot_path):
	
	metadata = pd.read_csv(meta_path, sep = '\t', header = None)
	metadata.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

	plots = []
	with open(plot_path, 'r') as f:
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

	x = df_stats[df_stats['#movies']<10]
	
	genres_to_remove = x['genre'].values.tolist()

	for word in genres_to_remove:
	    movies['genre_new'] = movies['genre_new'].apply(wordRemover(word).removeWord)
    
	movies_new = movies[~(movies['genre_new'].str.len() == 0)]

	ids = movies_new['movie_id'].values.tolist()

	reduced_genres = movies_new['genre_new'].values.tolist()

	binarizer_new = MultiLabelBinarizer()
	
	y_data_new = binarizer_new.fit_transform(reduced_genres)

	labels_tuple_list = []

	for i in range(len(ids)):
		labels_tuple_list.append((ids[i],y_data_new[i]))

	return ids, labels_tuple_list, y_data_new
