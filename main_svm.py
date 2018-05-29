import csv
import numpy as np
from sklearn import svm
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

''' 
quick sanity check here. I think it works
lables , data = readData('train.csv')
print(lables)
for i in range(5):
	print(data[i])
'''

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

labels, data = readTrainData('train.csv')
test = readTestData('test.csv')
print(linearPredict(labels, data, test))
print(svmPredict(labels, data, test))

''' What still needs to be done:
-Save the objects as files so that we don't have to fit the model every time we want to make predictions (use the pickle module for this)
-Check to see if these methods are even accurate. How do they compare to each other? Is computation time more important or is accuracy?
-write our output predictions to a csv for submission purposes 
'''