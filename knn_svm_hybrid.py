import csv
import numpy as np
import math
import statistics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def readTrainData(file):
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)	
	working_list = np.array(in_list)[1:].astype(int)
	labels = working_list[:,0]
	data = working_list[0:,1:]
	return data, labels

def testSample(labels, data, test_size):
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = test_size, random_state = 0)

	return data_train, data_test, labels_train, labels_test

def splitData(file):
	data, labels = readTrainData(file)
	count = int(math.floor(len(data)/10))
	data_test = []
	labels_test = []
	for i in range(0, count):
		data_test.append(data[i])
		labels_test.append(labels[i])
	data_train = []
	labels_train = []
	for i in range(count+1, len(data)):
		data_train.append(data[i])
		labels_train.append(labels[i])
	return data_test, labels_test, data_train, labels_train

def saveLinear(data, labels):
	for i in range(10):
		data_train, data_test, labels_train, labels_test = testSample(labels, data, 0.3)
		clf = svm.LinearSVC()
		clf.fit(data_train, labels_train)
		filename1 = 'trained_linear_%d.pkl'%(i,)
		filename2 = 'test_data_%d.pkl'%(i,)
		filename3 = 'test_labels_%d.pkl'%(i,)
		joblib.dump(clf, filename1)
		joblib.dump(data_test, filename2)
		joblib.dump(labels_test, filename3)

# function that loads a trained dataset that is saved in the working directory 
def loadTrainedSets():
	trained = []
	data = []
	labels = []
	for i in range(10):
		filename1 = 'trained_linear_%d.pkl'%(i,)
		filename2 = 'test_data_%d.pkl'%(i,)
		filename3 = 'test_labels_%d.pkl'%(i,)
		clf = joblib.load(filename1)
		test_data = joblib.load(filename2)
		test_labels = joblib.load(filename3)
		trained.append(clf)
		data.append(test_data)
		labels.append(test_labels)
	return trained, data, labels

def main():
	# reading in data, splitting it, fitting knn to the train data
	data_test, labels_test, data_train, labels_train = splitData('train.csv')
	neigh = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
	neigh.fit(data_train, labels_train)
	prediction_nn = neigh.predict(data_test)
	# gives a 2D array of the knn probabilities
	probs = neigh.predict_proba(data_test)
	# loading the saved linear svm sets in my working directory
	trained, data, labels = loadTrainedSets()
	# creating a 2D array of the predictions of all the saved linear svm sets
	prediction_pool = []
	for i in range(len(trained)):
		clf = trained[i]
		prediction_svm = clf.predict(data_test)
		prediction_pool.append(prediction_svm)
	# need the transpose for looping purposes
	prediction_pool = np.array(prediction_pool).T.tolist()
	final_prediction = []
	for i in range(len(probs)):
		high_odds = False
		for j in range(len(probs[i])):
			if probs[i, j] > 0.4:
				high_odds = True
		if high_odds:
			print('high odds')
			final_prediction.append(prediction_nn[i])
		else:
			print('low odds')
			try:
				most_frequent_number = statistics.mode(prediction_pool[i])
				final_prediction.append(most_frequent_number)
				print('trying')
			except statistics.StatisticsError:
				print('exception')
				final_prediction.append(prediction_nn[i])
	'''
	for i in range(len(prediction_pool)):
		nn_value = int(prediction_nn[i])
		try:
			most_frequent_number = statistics.mode(prediction_pool[i])
			if prediction_pool[i].count(most_frequent_number)>7:
				check = True
			else:
				check = False
			if nn_value != most_frequent_number and check:
				final_prediction.append(most_frequent_number)
			else:
				final_prediction.append(prediction_nn[i])
		except statistics.StatisticsError:
			final_prediction.append(prediction_nn[i])
	'''
	compare = []		
	for i in range(len(final_prediction)):
		if final_prediction[i] == labels_test[i]:
			compare.append(1)
		else:
			compare.append(0)
	print(np.mean(compare))
main()