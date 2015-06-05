#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation, grid_search
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', "salary", "bonus", "total_stock_value"] # You will need to use more features
features_list = ['poi', "fraction_to_poi", "bonus", "salary", "total_stock_value", 'exercised_stock_options']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
##Remove the outlier which was due to the summary of all the entries in the dataset
data_dict.pop("TOTAL",0)

# Scaling the features
def scale(finance_features):
    salary = []
    bonus = []
    total_stock_value = []
    exercised_stock_options = []
    fraction_to_poi = []
    for each in finance_features:
        fraction_to_poi.append(each[0])
        bonus.append(each[1])
        salary.append(each[2])
        total_stock_value.append(each[3])
        exercised_stock_options.append(each[4])
        
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    salary = scaler.fit_transform(salary)
    bonus = scaler.fit_transform(bonus)
    total_stock_value = scaler.fit_transform(total_stock_value)
    exercised_stock_options = scaler.fit_transform(exercised_stock_options)
    
    return zip(fraction_to_poi, bonus, salary, total_stock_value, exercised_stock_options)

### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == "NaN" or all_messages=="NaN":
        return 0
    else:
        fraction = float(poi_messages)/float(all_messages)
        
    return fraction


def add_poi_fraction(data_dict):
    
    for name in data_dict:
        data_point = data_dict[name]
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi
    
    
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi
    
    return data_dict 
     
data_dict = add_poi_fraction(data_dict)

#print data_dict

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#Call Feature sacling method

features = scale(features)

###Split the data test into training and testing set

train_features, test_features, train_labels, test_labels = cross_validation.train_test_split(features, labels, test_size =0.3, random_state=40)


parameters = { 'kernel': ('rbf', 'linear', 'poly', 'sigmoid'), "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "C" : [1,10, 100, 1000, 10000, 100000, 1000000] }


# Try the classifier
from sklearn.svm import SVC
svc = SVC()

#clf = SVC(C = 1000000, kernel = "rbf", gamma = 0.8)    # Provided to give you a starting point. Try a varity of classifiers.

#Using GridSearch
clf = grid_search.GridSearchCV(svc, parameters)


clf.fit(train_features,train_labels)
pred = clf.predict(test_features)

#Selecting best parameters using Gridserach 
print(clf.best_params_)
print 
#print(clf.grid_scores_)
print
#print "Accuracy::",accuracy_score(test_labels, pred)    
#print "Confusion matrix::", confusion_matrix(test_labels, pred)
#print "Classification Report::",classification_report(test_labels, pred)
print "Precision::",precision_score(test_labels, pred)
print "Recall::",recall_score(test_labels, pred)
