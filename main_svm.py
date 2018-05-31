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

# A funtion to fit LinearSVC to just a list of 4's and 9's
def train94(data, labels):
	labels_94 = []
	data_94 = []
	for i in range(len(labels)):
		if labels[i] == 9 or labels[i] == 4:
			labels_94.append(labels[i])
			data_94.append(data[i])
	#clf = svm.LinearSVC()
	#clf.fit(data_94, labels_94)

	return data_94, labels_94

# A function that attempts to separately distiguish between 9's and 4's
def twoStepPrediction(labels, data, test):
	# First we fit using every number
	whole = svm.LinearSVC()
	whole.fit(data, labels)
	# Fit a separate model using just 9's and 4's
	data_94, labels_94 = train94(data, labels)
	partial = svm.LinearSVC()
	partial.fit(data_94, labels_94)
	# make predictions based on all the numbers
	predictions_whole = whole.predict(test)
	final_predictions = []
	# if the whole model returns a prediction of a 9 or a 4, then check it against the partial model
	for i in range(len(predictions_whole)):
		if predictions_whole[i] == 4 or predictions_whole[i] == 9:
			prediction_partial = partial.predict([test[i]])
			final_predictions.append(prediction_partial[0])
		else:
			final_predictions.append(predictions_whole[i])
	return final_predictions

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

'''# Doing a little investigating to see what numbers give us the most problems
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
'''
# Comparing the simple Linear fit with the Two Step Linear Fit that focuses on 9's and 4's
def main():
	for i in range(10):
		y, X = readTrainData('train.csv')
		data_train, data_test, labels_train, labels_test = testSample(y, X, 0.4)
		prediction1 = linearPredict(labels_train, data_train, data_test)
		prediction2 = twoStepPrediction(labels_train, data_train, data_test)
		compare1 = []
		compare2 = []
		print('Run: ', i)
		for j in range(len(prediction1)):
			if prediction1[j] == labels_test[j]:
				compare1.append(1)
			else:
				compare1.append(0)
		print('Normal Prediction Accuracy:   ',np.mean(compare1))
		for k in range(len(prediction2)):
			if prediction2[k] == labels_test[k]:
				compare2.append(1)
			else:
				compare2.append(0)
		print('Two Step Prediction Accuracy: ', np.mean(compare2))
		print('----------------------------------')
main()

# I'm not seeing a real significant increase in accuracy, in some cases we see a decrease