import numpy as np
import pandas as pd
import cPickle as pickle
from textblob.classifiers import NaiveBayesClassifier

sms_raw = pd.read_csv('text.csv')

# training dataset 70%
# test dataset 30 %
sms_raw['split'] = np.random.randn(sms_raw.shape[0], 1)
msk = np.random.rand(len(sms_raw)) <= 0.7
train = sms_raw[msk]
test = sms_raw[~msk]

del train['split']
del test['split']

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

