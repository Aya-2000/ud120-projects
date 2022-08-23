#!/usr/bin/python

import sys
import pickle
from math import isnan
import os
import numpy as np
import pandas as pd
sys.path.append("../UD120-PROJECTS")

from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn import tree
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

"""
  NEW FEATURE DESCRIPTION
  milk - 
         (expenses + deferral payments) / 
         1 + (loan advances + long_term_incentive + deferred_income)
         POIs have an incentive to get as much out of the company as possible
         as they don't see a long-term future in the company. Expenses are
         consulting  and deferral payments
          must be paid by the company. Also Loan advances, long term incentives and deferred
         income must be paid by the company  later.
"""

features_list = ['poi','salary','exercised_stock',
                 'bonus',
                 'total_stock',
                 'fraction_of_deferred_income_to_total_payments',
                 'milk'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
for outlier in ['TOTAL','THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(outlier,0)
### Task 3: Create new feature(s)
for poi in data_dict:
    # NaN ---> 0
    for feature in ['expenses',
                    'deferral_payments',
                    'loan_advances',
                    'long_term_incentive',
                    'deferred_income',
                    'total_payments'
                   ]:
        if isnan(float(data_dict[poi][feature])):
            data_dict[poi][feature] = 0

    data_dict[poi]['milk'] = (data_dict[poi]['expenses'] +\
                              data_dict[poi]['deferral_payments']) / \
                             (1 + data_dict[poi]['loan_advances'] + \
                              data_dict[poi]['long_term_incentive'] + \
                              data_dict[poi]['deferred_income'])

    if data_dict[poi]['total_payments'] > 0:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = \
           data_dict[poi]['deferred_income'] / data_dict[poi]['total_payments']
    else:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import StratifiedShuffleSplit
def prec_recall (clf, data, feature_list, folds=1000):
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
     # true negative / false negative / true positive / false positive
    results = { 'tn': 0, 'fn': 0, 'tp': 0, 'fp': 0 }
    for train_idx, test_idx in cv:
        feature = { 'train': [], 'test': [] }
        label = { 'train': [], 'test': [] }
        for idx in train_idx:
            feature['train'].append(features[idx])
            label['train'].append(labels[idx])
        for idx in test_idx:
            feature['test'].append(features[idx])
            label['test'].append(labels[idx])
# fit classifier using training
        clf.fit(feature['train'], label['train'])
        predictions = clf.predict(feature['test'])
        for prediction, truth in zip(predictions, label['test']):
            if prediction == 0 and truth == 0: # T / Neg
                results['tn'] += 1
            elif prediction == 0 and truth == 1: # F / Neg
                results['fn'] += 1
            elif prediction == 1 and truth == 0: # F / Pos
                results['fp'] += 1
            elif prediction == 1 and truth == 1: # T / Pos
                results['tp'] += 1
            else:
                print ("Warning: Found a predicted label that's not 0 or 1")
                break

    try:
        precision = 1.0 * results['tp']/(results['tp']+results['fp'])
        recall = 1.0 * results['tp']/(results['tp']+results['fn'])
    except:
        print ("Got a divide by zero when trying out:", clf)

    return (precision, recall)        
# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
## DECISION TREE
#clf = tree.DecisionTreeClassifier()
## SUPPORT VECTOR MACHINE
# Scaler - for use with SVC and LinearSVC
#from sklearn.preprocessing import MinMaxScaler
#min_max_scalar = MinMaxScaler()
#data = min_max_scalar.fit_transform(data)
# LINEAR SUPPORT VECTOR MACHINE
#from sklearn.svm import LinearSVC
#clf = LinearSVC()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# GRID SEARCH
#from sklearn.grid_search import GridSearchCV
#svr = tree.DecisionTreeClassifier()
#clf = GridSearchCV(
#           svr,
#           {'criterion': ('gini','entropy'),
#            'splitter':  ('best','random'),
#            'max_features': [1,2,3,5,9,0.1,0.2,0.25,0.5,0.75,0.8,0.9,0.99,"auto","sqrt","log2",None]})
#            'max_features': [.8],
#            'class_weight': [None, "auto"],
#            'max_leaf_nodes': [None, 2,3,4,5,6,7,8,9,10],
#})
# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# TUNED RESULT
print( "\n","-"*34)
print (" Decision Tree Classifier Results")
print ("-"*34)
clf = tree.DecisionTreeClassifier(splitter="random")
precision, recall = prec_recall(clf, data, features_list)
print ("\nPrecision:", precision)
print ("   Recall:", recall)
print ("")

## Write out precision and recall values to results.csv to obtain statisical averages
#f = open('results.csv', 'w')
#f.write("Precision,Recall\n")
#
#precision_numbers = []
#recall_numbers = []
#for iteration in range(1000):
#    precision, recall = prec_recall(clf, data, features_list)
#    f.write("%f,%f\n"%(precision,recall))
#    precision_numbers.append(precision)
#    recall_numbers.append(recall)
#
#import numpy
#print ("                  PRECISION:                    ")
#print( " Mean:",numpy.mean(precision_numbers))
#print( " Median:",numpy.median(precision_numbers))
#print (" Maximum:",max(precision_numbers))
#print (" Minimum:",min(precision_numbers))
#print ("First Quartile:", numpy.percentile(precision_numbers, 25))
#print ("Third Quartile:", numpy.percentile(precision_numbers, 75))
#print ("  Mean:",numpy.mean(recall_numbers))
#print (" Median:",numpy.median(recall_numbers))
#print( " Maximum:",max(recall_numbers))
#print (" Minimum:",min(recall_numbers))
#print ("First Quart:", numpy.percentile(recall_numbers, 25))
#print ("Third Quart:", numpy.percentile(recall_numbers, 75))

####  review the performance of indivividual features 

#importances = clf.feature_importances_.tolist()
#print ("FEATURE IMPORTANCE")
#print ("Weight\tFeature")
#print ("------\t-------")
#for idx, feature in enumerate(features_list[1:]):
#    print( " %.2f \t%s" % (importances[idx], features)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)