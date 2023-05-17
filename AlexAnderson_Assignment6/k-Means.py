# Import relevant libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import read_csv
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import mixture

# Read the file
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = read_csv(url, names=names)

def re(dfp):
    _ = dfp.pop('clusters')
    m = dfp.describe().loc['mean']
    cols = dfp.columns
    for c in cols:
        dfp.loc[:,c] = dfp.loc[:,c].apply( lambda a: pow( (a-m[c]),2 ) )
    return(dfp.values.sum())
    
def reconErr(df):
    ire = [re(df[df['clusters'] == i]) for i in df['clusters'].unique()]
    return(sum(ire))

def kmeansModel(dat,n_cls):
    model = KMeans(n_clusters=n_cls, random_state=0)
    model.fit(dat)
    clusters = model.predict(dat)
    dat['clusters'] = clusters
    recon_error = reconErr(dat)
    return(recon_error)

#-----------------------------------------k-Means Clustering--------------------------------
#Create Arrays for Features and Classes
print("\n\nOutput:\n")
print("------------------------------------------------------------")
print('Part-1-k-Means Clustering')
print("------------------------------------------------------------")

y = data.pop('class')
X = data

plt.style.use('seaborn-whitegrid')

# Compute reconstruction error for k= 1 to 20
recon_errlist = [kmeansModel(X,i) for i in range(1,21)]

def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()

# Example where: recons_errlist is an array containing the reconstruction error for various values of k

plot_graph(recon_errlist, 'Reconstruction Error')

# Calculate the K algorithmically
dif = abs(np.diff(recon_errlist))

elbow_k=0
for i,val in enumerate(dif):
    if (i < (len(dif) - 1)):
        if( (val - dif[i+1]) > np.median(recon_errlist) ):
            elbow_k=elbow_k+1
print('Appropriate K from elbow curve:', elbow_k+1, '\n')

ek_model = KMeans(n_clusters=elbow_k, random_state=0)
ek_model.fit(X)
clusters = ek_model.predict(X)
k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

for k in np.unique(k_labels):
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]

print('Confusion matrix for k-means clustering using algorithmically generated elbow_k clusters:')
print(confusion_matrix(y, k_labels_matched))
print()
print('Accuracy score for k-means clustering using algorithmically generated elbow_k clusters:', accuracy_score(y, k_labels_matched), '\n')

b = data

k_3 = KMeans(n_clusters=3, random_state=0)
k_3.fit(b)
clusters = k_3.predict(b)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]

print('Confusion matrix for k-means clustering using k=3 clusters:')
print(confusion_matrix(y, k_labels_matched))
print('Accuracy score for k-means clustering using k=3 clusters:', accuracy_score(y, k_labels_matched), '\n')

#-----------------------------------------k-Means Clustering--------------------------------
#Create Arrays for Features and Classes
print("\n\nOutput:\n")
print("------------------------------------------------------------")
print('Part-2-GaussianModel')
print("------------------------------------------------------------")

b = data

def GaussianModel(dat,n_cls):
    gmm = mixture.GaussianMixture(n_components=n_cls, covariance_type="diag")
    gmm.fit(dat)
    
    return(gmm.aic(dat), gmm.bic(dat))

plt.style.use('seaborn-whitegrid')

# Compute aic and bic values for k= 1 to 20
aic_bic_list = [GaussianModel(b,i) for i in range(1,21)]
aic_list = [i[0] for i in aic_bic_list]
aic_elbow_k = 3

# Example where: recons_errlist is an array containing the reconstruction error for various values of k

plot_graph(aic_list, 'AIC')


bic_list = [i[1] for i in aic_bic_list]

plot_graph(bic_list, 'BIC')


# Calculate the K algorithmically
dif = abs(np.diff(bic_list))
bic_elbow_k=0
for i,val in enumerate(dif):
    if( val > np.subtract(*np.percentile(bic_list, [50, 25])) ):
        bic_elbow_k=bic_elbow_k+1
print('Appropriate K from elbow curve:', bic_elbow_k)

#GMM clustering using k=aic_elbow_k clusters
c = data

gmm_aic = mixture.GaussianMixture(n_components=3, covariance_type="diag")
gmm_aic.fit(c)
clusters = gmm_aic.predict(c)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
    
print('\nConfusion matrix for gmm clustering using k=aic_elbow_k clusters:\n')
print(confusion_matrix(y, k_labels_matched))
print('\nAccuracy score for gmm with k=aic_elbow_k clusters:', accuracy_score(y, k_labels_matched), '\n')

