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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#-----------------------------------------Naïve Baysian--------------------------------
#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test

#Encode for each class
array = preprocessing.LabelEncoder()
array.fit(y)
array.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

model = GaussianNB()
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
print("\n\nOutput:\n\n")
print("\n------------------------------------------------------------\n")
print("\nNaïve Baysian (GaussianNB): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1



#-----------------------------------------LINEAR REGRESSION--------------------------------
#Create Arrays for Features and Classes
arrayLR = dataset.values
X = arrayLR[:,0:4] #contains flower features (petal length, etc..)
y = arrayLR[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test


#Encode for each class
arrayLR = preprocessing.LabelEncoder()
arrayLR.fit(y)
arrayLR.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, arrayLR.transform(y), test_size=0.50, random_state=1)

model = LinearRegression()    
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nLinear regression (LinearRegression): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1


#-----------------------------------------POLYNOMIAL REGRESSION^2--------------------------------
#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test


#Encode for each class
array = preprocessing.LabelEncoder()
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
array.fit(y)
array.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_poly, array.transform(y), test_size=0.50, random_state=1)

model = LinearRegression()    
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nPolynomial of degree 2 regression (LinearRegression): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1


#-----------------------------------------POLYNOMIAL REGRESSION^3--------------------------------
#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names
#Split Data into 2 Folds for Training and Test


#Encode for each class
array = preprocessing.LabelEncoder()
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
array.fit(y)
array.transform(y)

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_poly, array.transform(y), test_size=0.50, random_state=1)

model = LinearRegression()    
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nPolynomial of degree 3 regression (LinearRegression) : \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1



#-----------------------------------------kNN--------------------------------
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

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nkNN (KNeighborsClassifier) : \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1


#-----------------------------------------LDA--------------------------------
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

model = LinearDiscriminantAnalysis()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nLDA (LinearDiscriminantAnalysis): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\

#-----------------------------------------QDA--------------------------------
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

model = QuadraticDiscriminantAnalysis()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nQDA (QuadraticDiscriminantAnalysis): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\

#-----------------------------------------svm.LinearSVC--------------------------------
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

model = svm.LinearSVC()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nSVM (svm.LinearSVC): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\


#-----------------------------------------Decision Tree--------------------------------
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

model = DecisionTreeClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nDecision Tree (DecisionTreeClassifiers): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\


#-----------------------------------------Random Forest--------------------------------
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

model = RandomForestClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nRandom Forrest (RandomForestClassifier): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\



#-----------------------------------------Extra Trees--------------------------------
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

model = ExtraTreesClassifier()
model.fit(X_Fold1, y_Fold1)
pred1 = model.predict(X_Fold2) #first fold testing
pred1 = pred1.round()
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing
pred2 = pred2.round()

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
predicted = predicted.round()
print("\n------------------------------------------------------------\n")
print("\nExtra Tree (ExtraTreesClassifier): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\

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
print("\n------------------------------------------------------------\n")
print("\nNN (neural_network.MLPClassifier): \n")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nClassification Report: \n")
print(classification_report(actual, predicted)) #P, R, & F1\