#!/usr/bin/python

import sys
import pickle
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import cross_validation, grid_search
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', "salary", "bonus", "total_stock_value", 'total_payments'] # You will need to use more features
#features_list = ['poi', "fraction_to_poi", "salary", "total_stock_value", 'exercised_stock_options']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("../final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
##Remove the outlier which was due to the summary of all the entries in the dataset
data_dict.pop("TOTAL",0)

        
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
     
my_dataset = add_poi_fraction(data_dict)

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)



###Split the data test into training and testing set
#train_features, test_features, train_labels, test_labels = cross_validation.train_test_split(features, labels, test_size =0.33, random_state=30)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import RandomForestClassifier

kf = cross_validation.KFold(len(features), n_folds = 5)

for train_index, test_index in kf:
        train_features = [features[ii] for ii in train_index]
        test_features = [features[ii] for ii in test_index]
        train_labels = [labels[ii] for ii in train_index]
        test_labels = [labels[ii] for ii in test_index]
                
#        clf = RandomForestClassifier(max_features = 'sqrt', min_samples_split = 2, criterion = 'gini', max_depth = 4, n_estimators = 27, random_state = 42)

    
        clf = RandomForestClassifier(n_estimators=27, random_state = 41)
        clf.fit(train_features,train_labels)
        pred = clf.predict(test_features)


test_classifier(clf, my_dataset, features_list)
#
#### Dump your classifier, dataset, and features_list so 
#### anyone can run/check your results.
#
dump_classifier_and_data(clf, my_dataset, features_list)






