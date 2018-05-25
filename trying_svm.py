
''' Trying the svm library in python '''
import csv
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[2., 2.]]))

# Some notes right away:
# The fit function takes support vectors. This should be easy for us because we are essentially given vectors in our dataset
# The outupt of the predict function is an array corresponding to what I think are the class labels. We will need to be able to interpret this in our context

'''Trying to convert csv to python lists '''
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

for i in range(1,len(your_list)):
	for j in range(1,len(your_list[i])):
		if your_list[i][j] == '0':
			your_list[i][j] = 0
		else:
			your_list[i][j] = 1 

for i in range(0,9):
	print(your_list[i])