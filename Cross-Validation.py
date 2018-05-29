#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
# Replace 'dat' with whatever your csv is called 
dat = datasets.load_iris()

### This first part just sets you up
dat.data.shape, dat.target.shape

X_train, X_test, y_train, y_test = train_test_split(dat.data, dat.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

# or wharever svm method that you use
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)


#This will get you 1 score, but 1 is not enough. We need an aggrigate score.
print(clf.score(X_test, y_test))



'''
for multiple runs (which i recomend) use this. cv is the number of times you
run it. I recomend as many as you can run quickly for your tests. 
Think of this kinda like bootstrapping. 
'''

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
scores = cross_val_score(clf, dat.data, dat.target, cv=5)
print(scores)
plt.hist(scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


''' 
this is just an example of histograms if the above does not work for your data
'''
rng = np.random.RandomState(10)  # deterministic random data
a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()