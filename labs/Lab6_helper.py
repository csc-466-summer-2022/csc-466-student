import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

# This is where we can get our models
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report

def scale(df):
    X = None
    ## BEGIN SOLUTION
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    ## END SOLUTION
    # YOUR SOLUTION HERE
    X = pd.DataFrame(X,columns=df.columns)
    return X

def pca(X,random_state=42):
    columns = ["Change me","Change me"]
    X_pca = None
    ## BEGIN SOLUTION
    pca = PCA(n_components=2,random_state=random_state)
    X_pca = pca.fit_transform(X)
    columns=["PC1 (%0.2f)"%pca.explained_variance_ratio_[0],"PC2 (%0.2f)"%pca.explained_variance_ratio_[1]]
    ## END SOLUTION
    # YOUR SOLUTION HERE
    X_pca = pd.DataFrame(X_pca,columns=columns)
    return X_pca

def kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10):
    kmeans_models = {}
    for n_clusters in range_n_clusters:
        ## BEGIN SOLUTION
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusterer.fit(X)
        kmeans_models[n_clusters] = clusterer
        ## END SOLUTION
        # Your solution here
        pass
    return kmeans_models

def assign_labels(X,kmeans_models):
    cluster_labels = {}
    for n_clusters in kmeans_models.keys():
        ## BEGIN SOLUTION
        cluster_labels[n_clusters] = kmeans_models[n_clusters].predict(X)
        ## END SOLUTION
        # Your solution here
        pass
    cluster_labels = pd.DataFrame(cluster_labels)
    return cluster_labels


def silhouette_scores(X,cluster_labels):
    def d(x,y): # For ease of use if you want it
        return np.sqrt(np.sum((x-y)**2))
    a = np.zeros((len(X),))
    b = np.zeros((len(X),))
    s = np.zeros((len(X),))
    ## BEGIN SOLUTION
    clusters = sorted(np.unique(cluster_labels))
    C = []
    for i in range(len(clusters)):
        C.append(np.where(cluster_labels == i)[0])
    for i in range(len(X)):
        c = cluster_labels[i]
        Xc = X.iloc[C[c],:]
        x = X.iloc[i,:]
        Xc = Xc.drop(x.name)
        distances = Xc.apply(lambda y: d(x,y),axis=1)
        a[i] = 1/(len(C[c])-1)*np.sum(distances)
        bs = []
        for j in range(len(clusters)):
            if j == c:
                continue
            Xj = X.iloc[C[j],:]
            bs.append(1/len(C[j])*np.sum(Xj.apply(lambda y: d(x,y),axis=1)))
        b[i] = np.min(bs)
        if a[i] < b[i]:
            s[i] = 1-a[i]/b[i]
        elif b[i] < a[i]:
            s[i] = b[i]/a[i]-1
    ## END SOLUTION
    # Your solution here
    print("First 10 a's")
    print(a[:10])
    print("First 10 b's")
    print(b[:10])
    return s

def bin_x(x,n_clusters=3,random_state=10):
    clusterer = None
    ## BEGIN SOLUTION
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(x)
    ## END SOLUTION
    # YOUR SOLUTION HERE
    return clusterer