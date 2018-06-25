import csv
import numpy as np
import math
import statistics
import smtplib
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def readTrainData(file):
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)	
	working_list = np.array(in_list)[1:].astype(int)
	labels = working_list[:,0]
	data = working_list[0:,1:]
	return data, labels

def readTestData(file):
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)
	working_list = np.array(in_list)[1:].astype(int)
	return working_list

def fitKNN():
	data, labels = readTrainData('train.csv')
	data_test = readTestData('test.csv')
	neigh = KNeighborsClassifier(n_neighbors=5, weights = 'distance', algorithm = 'brute')
	neigh.fit(data, labels)
	prediction_nn = neigh.predict(data_test)
	probs_nn = neigh.predict_proba(data_test)

	return prediction_nn, probs_nn

def fitRF():
	data, labels = readTrainData('train.csv')
	data_test = readTestData('test.csv')
	clf = RandomForestClassifier(n_estimators = 100)
	clf.fit(data, labels)
	prediction_rf = clf.predict(data_test)
	probs_rf = clf.predict_proba(data_test)

	return prediction_rf, probs_rf

def saveLinear(data, labels):
	for i in range(100):
		data_train, data_test, labels_train, labels_test = testSample(labels, data, 0.3)
		clf = svm.LinearSVC()
		clf.fit(data_train, labels_train)
		filename1 = 'trained_linear_%d.pkl'%(i,)
		filename2 = 'test_data_%d.pkl'%(i,)
		filename3 = 'test_labels_%d.pkl'%(i,)
		joblib.dump(clf, filename1)
		#joblib.dump(data_test, filename2)
		#joblib.dump(labels_test, filename3)

def loadTrainedSets():
	trained = []
	data = []
	labels = []
	for i in range(100):
		filename1 = 'trained_linear_%d.pkl'%(i,)
		clf = joblib.load(filename1)
		trained.append(clf)
	return trained

def writeToCSV(prediction):
	with open('output.csv', 'w') as myfile:
		wr = csv.writer(myfile)
		wr.writerow(prediction)

def main():
	data_test = readTestData('test.csv')
	print('Fitting Random Forest...')
	prediction_rf, probs_rf = fitRF()
	print(probs_rf)
	print('Fitting KNN...')
	prediction_nn, probs_nn = fitKNN()
	# loading my trained svm sets
	trained = loadTrainedSets()
	prediction_pool = []
	for i in range(len(trained)):
		clf = trained[i]
		prediction_svm = clf.predict(data_test)
		prediction_pool.append(prediction_svm)
	# need the transpose for looping
	prediction_pool = np.array(prediction_pool).T.tolist()
	final_prediction = []
	print('Making Predictions...')
	for i in range(len(prediction_nn)):
		max_prob_nn = max(probs_nn[i])
		max_prob_rf = max(probs_rf[i])
		index_max_prob_nn = list(probs_nn[i]).index(max_prob_nn)
		index_max_prob_rf = list(probs_rf[i]).index(max_prob_rf)
		try:
			max_prob_svm = statistics.mode(prediction_pool[i])
		except statistics.StatisticsError:
			max_prob_svm = prediction_rf[i]
			a = prediction_pool[i]
			d = {x:a.count(x) for x in a}

		possible_preds = [index_max_prob_nn, index_max_prob_rf, max_prob_svm]
		#try:
		#	final_prediction.append(statistics.mode(possible_preds))
		#except statistics.StatisticsError:
		#	final_prediction.append(index_max_prob_nn)
		if max_prob_nn > .5:
			final_prediction.append(prediction_nn[i])
		elif max_prob_rf > .75:
			final_prediction.append(prediction_rf[i])
		else:
			final_prediction.append(max_prob_svm)
	return final_prediction

writeToCSV(main())	