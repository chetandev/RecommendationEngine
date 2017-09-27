import os
import cPickle as pickle
import time
import sckittraining as train_model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

start_time = time.time()
cl = train_model.get_classifier()
predicted_svm = cl.predict(["Hello! We have credited Payment of Rs. 588.0 to your Vodafone mobile phone 9920183969. Your outstanding as on 2017-09-25 Rs. 0.19. Thank You.",
                        "Rakshita, we make a great team! We've counted your runs from last week & added Uber credits to your account. Check your app for more. T&C: t.uber.com/cric",
                        "Enjoy your weekend with Uber in Mumbai. Use code 'JOY50X3' & get 50% off (up to Rs 75) on 3 rides. Valid only on non uberPOOL rides until Sept 24. Ride now!"
                        ])
print((predicted_svm))

print("--- %s seconds ---" % (time.time() - start_time))