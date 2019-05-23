from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pickle

def train_test_svm(x_data, y_data):

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    """
    print "LinearSVC"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)
    print "LogisticRegression"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)
    """

    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
    ])

    parameters = [

        {'clf__estimator__kernel': ['rbf'],
         'clf__estimator__gamma': [1e-3, 1e-4],
         'clf__estimator__C': [0.01, 0.1, 1, 10],
         'clf__estimator__class_weight': ['balanced', None]
        },

        {'clf__estimator__kernel': ['poly'],
         'clf__estimator__C': [0.01, 0.1, 1, 10],
         'clf__estimator__class_weight': ['balanced', None]
        }
    ]

    grid_search(x_train, y_train, x_test, y_test, parameters, pipeline)


def grid_search(train_x, train_y, test_x, test_y, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print(classification_report(test_y, predictions))

with open('/Users/soumava/doc2vec_embeddings.pkl', 'rb') as handle:
    doc2vec = pickle.load(handle)

train_test_svm(doc2vec[0], doc2vec[1])

