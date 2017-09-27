import numpy as np
import pandas as pd
import cPickle as pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier





def train():
    data = pd.read_csv('text.csv')
    numpy_array = data.as_matrix()
    X = numpy_array[:, 0]
    Y = numpy_array[:, 1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3, n_iter=5, random_state=42)),
                             ])
    _ = text_clf_svm.fit(X_train, Y_train)

    print("model trained  ")

    predicted = text_clf_svm.predict(X_test)

    print("accuracy is : {}".format(np.mean(predicted == Y_test)))
    return text_clf_svm


def load_classifier():
    print('loading classifier')
    f = open('textclassifier.pickle', 'rb')
    cl = pickle.load(f)
    f.close()
    return cl

def save_classifier(cl):
    print('saving classifier')
    f = open('textclassifier.pickle', 'wb')
    pickle.dump(cl, f)
    f.close()

def get_classifier():
    if os.path.isfile('textclassifier.pickle'):
        return load_classifier()
    else:
        cl = train()
        save_classifier(cl)
        return cl



