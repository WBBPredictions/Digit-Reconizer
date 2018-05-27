import csv
import numpy as np
from sklearn import svm
''' This first function is to read in the data from a csv and put it into a list
takes a filename as an input '''

def readData(file):
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

# quick sanity check here. I think it works
lables , data = readData('train.csv')
print(lables)
for i in range(5):
	print(data[i])