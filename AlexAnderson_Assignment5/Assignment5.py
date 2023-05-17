import sys
import numpy as np
import pandas
from pandas import read_csv
import sklearn
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids 
from imblearn.under_sampling import TomekLinks 


#load dataset
url = "imbalanced iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#-----------------------------------------NN--------------------------------
#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test


#Encode for each class
array = preprocessing.LabelEncoder()
array.fit(y)
array.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array.transform(y), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n-----------------------------PART1-------------------------------\n")
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix
recall = recall_score(actual, predicted, average=None)
precision = precision_score(actual, predicted, average=None)
min1 = min(recall[0], precision[0])
min2 = min(recall[1], precision[1])
min3 = min(recall[2], precision[2])
cba = ((min1+min2+min3)/3)
print("\nClass Balanced Accuracy:\n")
print(cba)
sp1 = (cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])/((cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])+(cm[1,0]+cm[2,0]))
sp2 = (cm[0,0]+cm[0,2]+cm[2,0]+cm[2,2])/((cm[0,0]+cm[0,2]+cm[2,0]+cm[2,2])+(cm[0,1]+cm[2,1]))
sp3 = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])/((cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])+(cm[0,2]+cm[1,2]))
ba=(((sp1+sp2+sp3)/3)+((recall[0]+recall[1]+recall[2])/3))/2
print("\nBalanced Accuracy(From Lecture):\n")
print(ba)
print("\nBalanced Accuracy(From balanced_accuracy_score):\n")
print(balanced_accuracy_score(actual, predicted)) #balanced accuracy

print("\n-----------------------------PART2-------------------------------\n")
print("\nRebalancing using random oversampling\n")
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy="all")
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_over, array.transform(y_over), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix

print("\nRebalancing using SMOTE\n")
# define oversampling strategy
oversample = SMOTE(sampling_strategy="all")
# fit and apply the transform
X_sm, y_sm = oversample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_sm, array.transform(y_sm), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix

print("\nRebalancing using ADASYN\n")
# define oversampling strategy
oversample = ADASYN(sampling_strategy="minority")
# fit and apply the transform
X_ad, y_ad = oversample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_ad, array.transform(y_ad), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix

print("\n-----------------------------PART3-------------------------------\n")
print("\nRebalancing using random undersampling\n")
# define oversampling strategy
undersample = RandomUnderSampler(sampling_strategy="all")
# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_under, array.transform(y_under), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix

print("\nRebalancing using Clusters\n")
# define oversampling strategy
undersample = ClusterCentroids(sampling_strategy="all")
# fit and apply the transform
X_cl, y_cl = undersample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_cl, array.transform(y_cl), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix

print("\nRebalancing using Tomek\n")
# define oversampling strategy
undersample = TomekLinks(sampling_strategy="all")
# fit and apply the transform
X_tm, y_tm = undersample.fit_resample(X, y)
# summarize class distribution
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_tm, array.transform(y_tm), test_size=0.50, random_state=1)
model = MLPClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\nNN (neural_network.MLPClassifier):")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
cm = confusion_matrix(actual, predicted)
print(cm) #confusion matrix


