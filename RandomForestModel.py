"""
Author: Team Toronto
Date: 28/06/2017
Description: OPinion mining for the Yelp dataset
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as cPickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from collections import Counter
import matplotlib.pyplot as pyplot


def cleaningTheDataset(reviews):
    '''
        Cleaning the dataset 
        1. Tokenizing 
        2. Stop word removal
        3. Converting to lower case
        4. Removing all the special charecters
    '''
    reviews = re.sub('[^a-zA-Z]',' ',reviews)
    reviews = reviews.lower()
    reviews = reviews.split()
    reviews = [word for word in reviews if not word in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    
    return extractingPOP(reviews)

def extractingPOP(reviews):
    all_words = ""
    
    words = nltk.word_tokenize(reviews)
    tags = nltk.pos_tag(words)
    for each_tuple in tags:
        
        if each_tuple[1] == 'JJ' or each_tuple[1] == 'JJR' or each_tuple[1] == 'JJS' or each_tuple[1] == 'RB' or each_tuple[1] == 'RBR' or each_tuple[1] == 'RBS':
            all_words += each_tuple[0]+' '
            
    return all_words

def convertRatingsToLabel(ratings):
    
    R = ratings
    label = -1
    if R <= 2:
        label = 0
    elif 2.5 <= R <= 3.5:
        label = 10
    elif 4 <= R:
        label = 1
    
    else:
        label = 100
    print(label)
    return label
    


'''
    Importing the data set
'''

dataset = pd.read_csv('yelp_academic_dataset_review.csv')
print(dataset.columns)


print(dataset['text'].head())
ratings = []
for rating in dataset['stars'].iloc[:50000]:
    ratings.append(convertRatingsToLabel(rating))
categories = Counter(ratings)

print(categories)
label = []
for key in categories.keys():
    if key == 1:
        label.append("Positive")
    elif key == 0:
        label.append("Negative")
    else:
        label.append("Neutral")

        
"""
pyplot.axis("equal")
pyplot.pie(list(categories.values()),labels=label,autopct=None)
labels = [r'Positive (61.60 %)', r'Negative (20.64 %)', r'Neutral (17.74 %)']
plt.legend(labels, loc="best")
pyplot.show()

"""
#Preparing the vector space

#cv = CountVectorizer(ngram_range=(2,2),stop_words='english',lowercase=True,max_features=1000)
cv = TfidfVectorizer(ngram_range=(2,2),analyzer='word', stop_words='english',max_features=2000) 
dataset['text'] = dataset['text'].str.replace('\d+', '')
y =  pd.Series(ratings)


# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(dataset['text'].iloc[:50000], y, test_size = 0.20, random_state = 1)
x_train = cv.fit_transform(x_train).toarray()
print(cv.get_feature_names())
x_train = pd.DataFrame(x_train)
print(x_train.head())
print(x_train.shape)

x_test = cv.transform(x_test).toarray()

#Data and target for kfold cross validation
data = cv.fit_transform(dataset['text'].iloc[:50000]).toarray()
target = y
# Fitting Random forest to the Training set
#classifier = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# save the classifier
with open('random_forest.pkl', 'wb') as fid:
    cPickle.dump(classifier, fid)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""
print("Accuracy score is",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
kf_total = KFold(n_splits=10,shuffle=True)
acc =cross_val_score(classifier, data,target.ravel(),cv=kf_total)
print(acc.mean(),acc.std())

"""


