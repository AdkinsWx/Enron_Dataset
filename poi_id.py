
# coding: utf-8

# # Enron Machine Learning Project
# ### By Matthew Adkins
# 
# In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.
# 
# In this project I will apply my knowledge of machine learning and build a person of interest identifier. The identifier will harness financial and email data that was made public.
# 
# There are four critical steps in this project:
# 
# 1. Enron Dataset Cleaning
# 2. Feature Creation and Processing
# 3. Applying a Learning Algorithm
# 4. Validating the Accuracy of the Algorithm

# ## 1. The Enron Dataset
# 
# 
# We'll start by loading the loading up the dataset and then we'll take a peak at what we can use to start creating an effective identifier.

# In[1]:

import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "poi_per_to_msg", "poi_per_from_msg",'bonus', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# Now that we have the data loaded in to our programs let's take a look at the structure.
# 
# We'll see how many executives are included in the set as well as what kind of financial and email data we have on each one.

# In[2]:

print "Total executives: ", len(data_dict.keys())


# In[3]:

print data_dict['METTS MARK']


# So, we can see that there are 146 executives in the data set with numerous amounts of information on each. While we are looking at the features in this data we should think about what features would be helpful in creating new features from. 
# 
# Originally I made two features options_per_stock and bonus_per_salary. The logic was that person's of interest (poi) would sell their equity in the company and pay themselves outsized bonuses. However, I could not obtain a high accuracy with the correct precision and recall parameters. So, in this iteration of the project we will look at two new features:
# 
# 1. poi_per_to_msg
# 2. poi_per_from_msg
# 
# 
# These two features look at an individuals to and from messages and what fraction of each where to known poi's. Since the the financial behaviors were harder to pin down let's see if we have better luck using email behaviors.
# 
# 
# 
# 
# 
# Now before we get too far let's look for any obvious outliers. We'll plot and see if we can visually see any of the outliers.

# In[4]:

### Task 2: Remove outliers

#Plot showing outlier
features=['bonus', 'salary']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary,bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()


# After plotting the graph we can clearly see an outlier. This outlier happens to be the total salary for the whole dataset. So, since it is not an actual person we will remove it leaving us with 145 real Enron executives.
# 
# While we are removing the outliers we will also remove any NaN's from the data set.

# In[5]:

### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))


#plot after removing the outlier
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary,bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()


# After removing the outlier we can see the graph is much easier to interpret. There does seem to be a few outliers but these may also be our poi's.
# 
# 
# 
# 
# ## 2. Feature Processing
# 
# Now that we have finished cleaning and removing outliers from the we can create two new features. 
# 
# We will use the exercised_stock_options and total_stock_value to create options_per_stock. This will explore possible poi's that were dumping their Enron equity.
# 
# We will also use bonus and salary to find anyone that received an outsized bonus and try to concluded if that makes them a poi.

# In[6]:

### Task 3: Create new feature(s)

### create new features
### bonus_per_salary, options_per_stock

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### creating new list of features
poi_per_to_msg=dict_to_list("from_poi_to_this_person","to_messages")
poi_per_from_msg=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["poi_per_to_msg"]=poi_per_to_msg[count]
    data_dict[i]["poi_per_from_msg"]=poi_per_from_msg[count]
    count +=1


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)

## split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)


#Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)


# ## 3. Applying Learning Algorithms
# 
# 
# Now that we created our new features that we want to explore lets take a look at how accurate our creations are at identifying poi's.

# In[7]:

t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print "Naive Bayes accuracy score: " , accuracy

print "Naive Bayes training time:", round(time()-t0, 3), "s"



# Using the Naive Bayes algorithm we can an accuracy rating of 0.72 but let's try another algorithm to see if we can obtain a better accuracy.
# 
# 

# In[8]:

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print 'Decision Tree accuracy: ', accuracy

print "Decision tree training time:", round(time()-t0, 3), "s"


# After attempting a DecisionTree alogrithm it looks like our accuracy is significantly better from when we used NaiveBayes. 
# But let's see if we can tune the DecisionTree algorithm to get an even higher accuracy score. We will manipulate the min_samples_split parameter to attempt for better results.
# 

# ## 4. Validating and Algorithm Tuning
# 
# Here we will take DecisionTree algorithm and tune it to see if we improve on our already superior results.

# In[13]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


t0 = time()

clf = DecisionTreeClassifier(min_samples_split = 6)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Algorithm validation:"

print "Decision tree training time:", round(time()-t0, 3), "s"
# function for calculation ratio of true positives
# out of all positives (true + false)
print 'precision = %0.3f' % precision_score(labels_test, pred)


# function for calculation ratio of true positives
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test, pred)


print "accuracy after tuning = ", accuracy

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# After some tuning and setting the min_samples_split in our DecisionTree to 6 we were able to raise our accuracy to 0.91 and achieve an accuracy and recall of 0.67 and 1.0 respectively.

# ## Conclusions
# 
# Interpretating our precision an recall values helps us determine how well our program is at identifying POI's from other executives. Precision measures false positives or how many executives would be falsely identified as POI's. Given our value of 0.67 this means that 33% of our identified POI's are false alarms. 
# 
# Our other measure recall measures false negatives or how many POI's would be misidentified as regular executives. Given our value of 1.0 this means that we have captured all of the POI's but at the price of misidentifying regular executives. This could be a case of overfitting the data and thus capturing too many points that don't belong in the POI group.
# 
# As mentioned in the beginning of this paper I attempted to use financial data but the precision and recall was far below our threshold value of 0.3. This means that using the financial parameters I created was very inefficient. With this said further looking into the actual words used in the emails to/from the POI's could help keep our 33% of innocent executives safe from prosecution. Using a data set of keywords used between executives and the POI's would make our algorithm even more accurate.
