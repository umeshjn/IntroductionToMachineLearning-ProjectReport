#!/usr/bin/python
#This code is used for selecting the 5 best features out of all the financial features along 
# with the two newly created features created using the email features. 

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', "fraction_from_poi", "fraction_to_poi", 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features
features_list = ['poi',  "fraction_to_poi", 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

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
     
data_dict = add_poi_fraction(data_dict)

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



###Split the data test into training and testing set
train_features, test_features, train_labels, test_labels = cross_validation.train_test_split(features, labels, test_size =0.3, random_state=40)

#Feature selection
ch2 = SelectKBest(k=8)
train_features = ch2.fit_transform(train_features, train_labels)
test_features = ch2.transform(test_features)

print ch2.get_support()











