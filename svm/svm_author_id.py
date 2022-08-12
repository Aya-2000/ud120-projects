#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../UD120-PROJECTS/tools/..")
import joblib
data_dict = joblib.load( open("../UD120-PROJECTS/tools/email_authors.pkl", "rb" ))
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
def submitAccuracy():
    return acc
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
