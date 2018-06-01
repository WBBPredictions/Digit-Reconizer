from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import csv

# using the data reader from the svm file
def readTrainData(file):
	with open(file, 'r') as f:
		reader = csv.reader(f)
		in_list = list(reader)	
	working_list = np.array(in_list)[1:].astype(int)
	labels = working_list[:,0]
	data = working_list[0:,1:]
	return data, labels

# also stealing this function for testing 
def testSample(labels, data, test_size):
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = test_size)

	return data_train, data_test, labels_train, labels_test

# attempting to take 5 random samples and test accuracy
for i in range(5):
	X, y = readTrainData('train.csv')
	data_train, data_test, labels_train, labels_test = testSample(y, X, 0.3)
	neigh = KNeighborsClassifier(n_neighbors=4, weights = 'distance')
	neigh.fit(data_train, labels_train)
	prediction = neigh.predict(data_test)
	compare = []
	for j in range(len(prediction)):
		if prediction[j] == labels_test[j]:
			compare.append(1)
		else:
			compare.append(0)
	print(np.mean(compare))