import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
''' This first function is to read in the data from a csv and put it into a list
takes a filename as an input '''

def readTrainData(file):
	# reading the csv file in as strings and converting them to integers
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)
	# turning the raw csv list into a list of integers with the headers removed	
	working_list = np.array(in_list)[1:].astype(int)
	labels = working_list[:,0]
	data = working_list[0:,1:]
	# returns a tuple of values
	# label is the number seen in the image
	# data is the pixel data for each image
	return labels, data

''' This function reads the test data and returns in as a python array '''

def readTestData(file):
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)
	working_list = np.array(in_list)[1:].astype(int)
	return working_list

''' This function fits an svm model to a matrix of data with a vector of labels.
Then it makes predictions based on the test data.
The goal is to have it output a vector of predicted numbers.
This will theoretically work, but with our data size, it will be computationally expensive.
I think the best opotions will be to shrink our data, or research LinearSVC. http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
'''

def svmPredict(labels, data, test):
	# creating an svc object
	# Empty parentheses create the object with the default parameters
	clf = svm.SVC()
	# fitting the model 
	clf.fit(data, labels)
	# creating a list of predicted values
	prediction = clf.predict(test)

	return prediction

''' Using a linear approach to decrease run time. Runs relatively fast considering our data size. No idea if it's accurate '''
def linearPredict(labels, data, test):
	# creating the LinearSVC object
	clf = svm.LinearSVC()
	# fitting the model to our data
	clf.fit(data, labels)
	# making predictions based on test data
	prediction = clf.predict(test)

	return prediction

''' Adding a cross validation function '''
def testSample(labels, data, test_size):
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = test_size, random_state = 0)

	return data_train, data_test, labels_train, labels_test

# Trying out different parameters to see if we get any drastic changes
# Smaller C vlaue seems to work slightly better although this may be due to sampling variation
''' 
y, X = readTrainData('train.csv')
params = [0.001, 0.01, 0.05, 0.1, 1, 5, 10]
for i in range(len(params)):
	score_list = []
	for j in range(10):
		data_train, data_test, labels_train, labels_test = testSample(y, X, 0.4)
		clf = svm.LinearSVC(C = params[i])
		clf.fit(data_train, labels_train)
		score_list.append(clf.score(data_test, labels_test))
	print(score_list)
	print(np.mean(score_list))
'''

# Doing a little investigating to see what numbers give us the most problems
# Will print out the model accuracy, then predicted value in the first column, actual value in the second column
y, X = readTrainData('train.csv')
data_train, data_test, labels_train, labels_test = testSample(y, X, 0.4)
clf = svm.LinearSVC(C = 0.01)
clf.fit(data_train, labels_train)
prediction = clf.predict(data_test)
print(clf.score(data_test,labels_test))
for i in range(len(prediction)):
	if prediction[i] != labels_test[i]:
		print(prediction[i],labels_test[i])

# Looks like the system is confusing 9's and 4's a lot.
# I suggest we look into a K-nearest neighbors approach with weights on the list of 9's and 4's
# I think this will push us above 90% accuracy