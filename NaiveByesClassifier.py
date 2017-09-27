# import numpy as np
# import pandas as pd
# import os.path
# import cPickle as pickle
# from textblob.classifiers import NaiveBayesClassifier
#
#
#
# if os.path.isfile('textclassifier.pickle'):
#      print('in')
#      f=  open('textclassifier.pickle', 'rb')
#      cl = pickle.load(f)
#      f.close()
# else:
#      print('out')
#      with open('train.csv', 'r') as fp:
#         cl = NaiveBayesClassifier(fp, format="csv")
#      f = open('textclassifier.pickle', 'wb')
#      pickle.dump(cl,f)
#      f.close()
#
#
#
# # with open('test.csv','r') as f:
# #     a = cl.accuracy(f,format="csv")
# #
# # print(a)
#
#
# # cl.show_informative_features(20)
#
#
#
#
#
#
# print(cl.classify("Hello! We have credited Payment of Rs. 588.0 to your Vodafone mobile phone 9920183969. Your outstanding as on 2017-09-25 Rs. 0.19. Thank You."))
#
# print(cl.classify("Rakshita, we make a great team! We've counted your runs from last week & added Uber credits to your account. Check your app for more. T&C: t.uber.com/cric"))
#
# print(cl.classify("Enjoy your weekend with Uber in Mumbai. Use code 'JOY50X3' & get 50% off (up to Rs 75) on 3 rides. Valid only on non uberPOOL rides until Sept 24. Ride now!"))









