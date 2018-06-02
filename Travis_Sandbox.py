#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pa

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


def main():
    x, y = readTrainData('Mini_train.csv')
    x2 = pa.DataFrame(data =x)
    #x2.DataFrame.
    print(x)
    
main()
    
