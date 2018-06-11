import csv
import numpy as np
import math
import statistics
import smtplib
from sklearn import svm
from sklearn.externals import joblib
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
	for i in range(30):
		data_train, data_test, labels_train, labels_test = testSample(labels, data, 0.3)
		clf = svm.LinearSVC()
		clf.fit(data_train, labels_train)
		filename1 = 'trained_linear_%d.pkl'%(i,)
		filename2 = 'test_data_%d.pkl'%(i,)
		filename3 = 'test_labels_%d.pkl'%(i,)
		joblib.dump(clf, filename1)
		#joblib.dump(data_test, filename2)
		#joblib.dump(labels_test, filename3)

# function that loads a trained dataset that is saved in the working directory 
def loadTrainedSets():
	trained = []
	data = []
	labels = []
	for i in range(30):
		filename1 = 'trained_linear_%d.pkl'%(i,)
		#filename2 = 'test_data_%d.pkl'%(i,)
		#filename3 = 'test_labels_%d.pkl'%(i,)
		clf = joblib.load(filename1)
		#test_data = joblib.load(filename2)
		#test_labels = joblib.load(filename3)
		trained.append(clf)
		#data.append(test_data)
		#labels.append(test_labels)
	return trained

def sendEmail(msg):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login("WBBPredictions@gmail.com", "Team6969")
	server.sendmail("WBBPredictions@gmail.com", "terrisbecker@gmail.com", msg)
	server.quit()

def main():
	# reading in data, splitting it, fitting knn to the train data
	print('Reading data...')
	data_test, labels_test, data_train, labels_train = splitData('train.csv')
	# loading the saved linear svm sets in my working directory
	print('Loading trained sets...')
	trained = loadTrainedSets()
	# fitting the knn model and making predictions
	print('Fitting KNN...')
	neigh = KNeighborsClassifier(n_neighbors=4, weights = 'distance', algorithm = 'brute')
	neigh.fit(data_train, labels_train)
	prediction_nn = neigh.predict(data_test)
	# gives a 2D array of the knn probabilities
	probs = neigh.predict_proba(data_test)
	# creating a 2D array of the predictions of all the saved linear svm sets
	prediction_pool = []
	for i in range(len(trained)):
		clf = trained[i]
		prediction_svm = clf.predict(data_test)
		prediction_pool.append(prediction_svm)
	# need the transpose for looping purposes
	prediction_pool = np.array(prediction_pool).T.tolist()
	final_prediction = []
	print('Making predictions...')
	for i in range(len(probs)):
		high_odds = False
		for j in range(len(probs[i])):
			if probs[i, j] > 0.4:
				high_odds = True
		if high_odds:
			final_prediction.append(prediction_nn[i])
		else:
			try:
				most_frequent_number = statistics.mode(prediction_pool[i])
				final_prediction.append(most_frequent_number)
			except statistics.StatisticsError:
				final_prediction.append(prediction_nn[i])
	compare = []		
	for i in range(len(final_prediction)):
		if final_prediction[i] == labels_test[i]:
			compare.append(1)
		else:
			compare.append(0)
	print('Done...')
	result = str(np.mean(compare))
	print(result)
	sendEmail(result)
main()