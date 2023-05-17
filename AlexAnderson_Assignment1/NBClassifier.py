# load libraries
import numpy as np
import scipy
import sklearn
import sys
import pandas
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

model = GaussianNB()
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testin

irisSetosa = 1
irisVersicolour = 2
irisVirginica = 3

#Encode for each class
array = preprocessing.LabelEncoder()
array.fit(y)
array.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array.transform(y), test_size=0.50, random_state=1)

LR = LinearRegression()    
LR.fit(X_Fold1, y_Fold1) #first fold training
pred1 = LR.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
LR.fit(X_Fold2, y_Fold2) #second fold training
pred2 = LR.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1