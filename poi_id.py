#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import pickle
import numpy as np
sys.path.append("../tools/")
import pprint
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, decomposition
from sklearn.naive_bayes import GaussianNB

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'expenses', 'total_stock_value', 'bonus', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

POI_label = ['poi']

total_features = POI_label + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Total number of data points = ', len(data_dict)

# allocation across classes (POI/non-POI)
poi_count = 0
for employee in data_dict:
    if data_dict[employee]['poi'] == True:
        poi_count += 1
print 'number of POI = ', poi_count
print 'number of non-POI = ', len(data_dict) - poi_count

# number of features used
print 'total number of available features for every employee = ', len(total_features), 'which are: ', total_features
print 'number of features used = ', len(features_list), 'which are: ', features_list

# are there features with many missing values? etc.
missing_values = {}
for feat in total_features:
    missing_values[feat] = 0

for emp in data_dict:
    for f in data_dict[emp]:
        if data_dict[emp][f] == 'NaN':
            missing_values[f] += 1
            # fill NaN values
            # data_dict[emp][f] = 0

print 'missing values: ', missing_values

### Task 2: Remove outliers

def show_scatter_plot(dataset, feature1, feature2):
    """ given two features feature1 (x) and feature2 (y),
    this function creates a 2D scatter plot showing
    both x and y
    """
    data = featureFormat(dataset, [feature1, feature2])
    for p in data:
        x = p[0]
        y = p[1]
        plt.scatter(x, y)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

# identify outliers
show_scatter_plot(data_dict, "salary", "bonus")

# remove them
data_dict.pop( "TOTAL", 0 )
data_dict.pop("FREVERT MARK A", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

show_scatter_plot(data_dict, "salary", "bonus")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# create new features
def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
    """
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)

    return fraction

def takeSecond(elem):
    """ take second element for sort
    """
    return elem[1]

for emp in my_dataset:
    from_poi_to_this_person = my_dataset[emp]['from_poi_to_this_person']
    to_messages = my_dataset[emp]['to_messages']
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    # print fraction_from_poi
    my_dataset[emp]['fraction_from_poi'] = fraction_from_poi

    from_this_person_to_poi = my_dataset[emp]['from_this_person_to_poi']
    from_messages = my_dataset[emp]['from_messages']
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    my_dataset[emp]['fraction_to_poi'] = fraction_to_poi

features_list_n = total_features
features_list_n.remove('email_address')
features_list_n =  features_list_n + ['fraction_from_poi', 'fraction_to_poi']
print features_list_n

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)

# univariate feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 10)
selector.fit(features, labels)
scores = zip(features_list_n[1:], selector.scores_)
sorted_scores = sorted(scores, key = takeSecond, reverse = True)
print 'SelectKBest scores: ', sorted_scores

kBest_features = POI_label + [(i[0]) for i in sorted_scores[0:11]]
kBest_features.remove('fraction_to_poi')
print 'KBest', kBest_features

for emp in data_dict:
    for f in data_dict[emp]:
        if data_dict[emp][f] == 'NaN':
            # fill NaN values
            data_dict[emp][f] = 0

my_dataset = data_dict


# dataset without new features
from sklearn import preprocessing
data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# dataset with new features
kBest_new_features = kBest_features + ['fraction_from_poi', 'fraction_to_poi']
data = featureFormat(my_dataset, kBest_new_features, sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# split 30% of the data for testing
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test =      cross_validation.train_test_split(features, labels, test_size=0.3, random_state=43)

from sklearn.metrics import accuracy_score, precision_score, recall_score

             
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from time import time
clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
pred = clf1.predict(features_test)
naivebayes_score = clf1.score(features_test, labels_test)

naivebayes_acc = accuracy_score(labels_test, pred)
naivebayes_pre = precision_score(labels_test, pred)
naivebayes_rec = recall_score(labels_test, pred)

print "Naive Bayes accuracy: ", naivebayes_acc
print "Naive Bayes precision: ", naivebayes_pre
print "Naive Bayes recall: ", naivebayes_rec

from sklearn.svm import SVC
clf5 = SVC( kernel = 'linear', C = 0.1, gamma = 1)
clf5.fit(features_train, labels_train)
pred = clf5.predict(features_test)
svm_score = clf5.score(features_test, labels_test)

svm_acc = accuracy_score(labels_test, pred)
svm_pre = precision_score(labels_test, pred)
svm_rec = recall_score(labels_test, pred)

print "SVM accuracy: ", svm_acc
print "SVM precision: ", svm_pre
print "SVM recall: ", svm_rec

from sklearn import tree
clf2 = tree.DecisionTreeClassifier()
clf2.fit(features_train, labels_train)
pred = clf2.predict(features_test)
decisiontree_score = clf2.score(features_test, labels_test)

decisiontree_acc = accuracy_score(labels_test, pred)
decisiontree_pre = precision_score(labels_test, pred)
decisiontree_rec = recall_score(labels_test, pred)
print "Decision Tree accuracy: ", decisiontree_acc
print "Decision Tree precision: ", decisiontree_pre
print "Decision Tree recall: ", decisiontree_rec

from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=10)
clf3.fit(features_train, labels_train)
pred = clf3.predict(features_test)
randomforest_score = clf3.score(features_test, labels_test)

randomforest_acc = accuracy_score(labels_test, pred)
randomforest_pre = precision_score(labels_test, pred)
randomforest_rec = recall_score(labels_test, pred)
print "Random Forest accuracy: ", randomforest_acc
print "Random Forest precision: ", randomforest_pre
print "Random Forest recall: ", randomforest_rec

from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression(C=1e5)
clf4.fit(features_train, labels_train)
pred = clf4.predict(features_test)
logisticr_score = clf4.score(features_test, labels_test)

logisticr_acc = accuracy_score(labels_test, pred)
logisticr_pre = precision_score(labels_test, pred)
logisticr_rec = recall_score(labels_test, pred)
print "Logistic Regression accuracy: ", logisticr_acc
print "Logistic Regression precision: ", logisticr_pre
print "Logistic Regression recall: ", logisticr_rec

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def tune_params(grid_search, features, labels, params, iters = 80):
    
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predicts)] 
        pre = pre + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
    print "accuracy: {}".format(np.mean(acc))
    print "precision: {}".format(np.mean(pre))
    print "recall: {}".format(np.mean(recall))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


from sklearn.model_selection import GridSearchCV

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(estimator = nb_clf, param_grid = nb_param)
print("Naive Bayes model evaluation")
tune_params(nb_grid_search, features, labels, nb_param)
tune_params(nb_grid_search, new_features, new_labels, nb_param)


# SVM
from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

print("SVM model evaluation")
tune_params(svm_grid_search, features, labels, svm_param)
tune_params(svm_grid_search, new_features, new_labels, svm_param)


# Decision Tree
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

print("Decision Tree model evaluation")
tune_params(dt_grid_search, features, labels, dt_param)
tune_params(dt_grid_search, new_features, new_labels, dt_param)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)
rf_param = {}
rf_grid_search = GridSearchCV(estimator = rf_clf, param_grid = rf_param)

print("Random Forest model evaluation")
tune_params(rf_grid_search, features, labels, rf_param)
tune_params(rf_grid_search, new_features, new_labels, rf_param)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_param = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 0.01, 0.001, 0.0001]}
lr_grid_search = GridSearchCV(estimator = lr_clf, param_grid = lr_param)

print("Logistic Regression model evaluation")
tune_params(lr_grid_search, features, labels, lr_param)
tune_params(lr_grid_search, new_features, new_labels, lr_param)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=43)

from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
final_features_list = kBest_new_features



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




