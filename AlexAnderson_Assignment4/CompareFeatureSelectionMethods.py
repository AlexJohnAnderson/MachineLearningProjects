# load libraries
import numpy as np
import scipy
import sklearn
import sys
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from numpy.linalg import eig

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

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

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

model = DecisionTreeClassifier()
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
print("\n\nOutput:\n")
print("------------------------------------------------------------")
print("\nPart 1: ")
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nFeatures: \n")
print(names)
print('\n\n')

#-----------------------------------------Decision Tree 2--------------------------------
#Create Arrays for Features and Classes
# Create PCA instance
pca = PCA(n_components=4)
# Perform PCA
pca.fit(X)
# Get eigenvectors and eigenvalues
vectors = pca.components_
values = pca.explained_variance_

# Transform data
principleComponents = pca.transform(X)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = list(zip(values, vectors))
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda X: X[0], reverse=True)
# Transform data (X) to Z
W = eigen_pairs[0][1].reshape(4, 1)
Z = principleComponents.dot(W)

# Proportion of variance
L1 = 4.24025608 
SumL = np.sum(values)
pov = L1/SumL

X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(Z, y, test_size=0.50, random_state=1)

model = DecisionTreeClassifier()
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([pred1, pred2]) #predicted classes
print("\n\nOutput:\n")
print("------------------------------------------------------------")
print("\nPart 2: ")
print("\nEigenvalues: \n")
print(values)
print("\nEigenvectors: \n")
print(vectors)
print("\nPoV: \n")
print(pov)
print("\nAccuracy: \n")
print(accuracy_score(actual, predicted)) #accuracy
print("\nConfusion Matrix: \n")
print(confusion_matrix(actual, predicted)) #confusion matrix
print("\nFeatures: \n")
print("[z1, Z2, Z3, Z4]")

#-----------------------------------------Decision Tree 3--------------------------------
def modelsvm(df,label):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.5, random_state=1)
        #Derive two folds for cross validation
        X_tr1 = X_train; y_tr1 = y_train
        X_tst1 = X_test; y_tst1 = y_test

        X_tr2 = X_test; y_tr2 = y_test
        X_tst2 = X_train; y_tst2 = y_train

        from sklearn.svm import SVC
        m = SVC(gamma=.1, kernel='linear', probability=True)

        # fold1
        clf = m.fit(X_tr1,y_tr1)
        y_pred1 = clf.predict(X_tst1)

        # fold2
        clf = m.fit(X_tr2,y_tr2)
        y_pred2 = clf.predict(X_tst2)

        # Inverse transform
        y_raw   = [*y_pred1, *y_pred2]
        y_predr = le.inverse_transform(y_raw)
        y_testr = le.inverse_transform([*y_tst1, *y_tst2])
        
        # Metrics
        acc = accuracy_score(y_testr, y_predr)
        cmsv=confusion_matrix(y_testr, y_predr)
        
        return acc, cmsv 
    except Exception as ex:
           print ("Exception occured in svm model-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
pca_df = pd.DataFrame(W.T, columns=['sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca'])
icol = ['sepal-length','sepal-width','petal-length','petal-width','sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca'] 
iris_pca = np.vstack((X, pca_df)) 

# Simulated annealing
import random
iters = 100
accepted_accuracy = 0
accepted_subset = []
current_subset = icol #random.sample(icol,2)
stat = ' '; rr=0; Pr_accept = 0
restart_counter= 0
best_accuracy= 0
best_feature_subset= current_subset
##random.seed(23)

for i in range(iters):
    print('Iteration:',i)
    if(random.choice([0,1]) == 0):#add
        
        # Features excluding the current subset
        nodup_list = list( set(icol).difference(current_subset) )
        
        if (len(nodup_list) > 1):
            plist =random.sample(nodup_list, random.choice([1,2]) )
            current_subset = current_subset + plist
            
        elif (len(nodup_list) == 1):
            plist = nodup_list
            current_subset = current_subset + plist
            
        else: #current subset has all the features, so remove random features
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = list( set(current_subset).difference(plist) )       
        
    else:#delete
    
        if (len(current_subset) == 0): #if current subset is empty, add randome features
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = current_subset + plist
        else:
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = list( set(current_subset).difference(plist) )
    
    # discard the empty set
    if (len(current_subset) == 0):
        print('Empty set - discard')
        print('---------------------------------------------')
        continue
        
    temp_df = iris_pca[i]
    model_acc, cmsv = modelsvm(temp_df ,y)
    
    if (accepted_accuracy < model_acc):
        accepted_accuracy = model_acc
        stat = 'Improved'
    else:
        Pr_accept = np.exp( -1 * i * ( (accepted_accuracy - model_acc) / accepted_accuracy ) )
        rr = np.random.random()
        if (rr > Pr_accept):
            current_subset = accepted_subset #reject the new dataset
            stat = 'Reject'
        else:
            accepted_accuracy = model_acc
            stat = 'Accept'
    
    # Restart logic
    if (model_acc > best_accuracy):
        best_accuracy = model_acc
        best_feature_subset = current_subset
        restart_counter= 0
    else:
        restart_counter = restart_counter + 1
        if (restart_counter == 10):
            current_subset = best_feature_subset
            model_acc = best_accuracy
            restart_counter= 0
            stat = 'Restart'
            
    accepted_subset = current_subset
    accepted_accuracy = model_acc
    
    print('Accepted Features:',accepted_subset)
    print('Accuracy:',model_acc)
    print('Pr[accept]:',Pr_accept)
    print('Random uniform:',rr)
    print('Status:',stat)
    print('---------------------------------------------')

part3_features = accepted_subset
print('Confusion Matrix - Simulated annealing:')
print(cmsv)



