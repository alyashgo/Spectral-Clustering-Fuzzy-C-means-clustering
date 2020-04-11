# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:48:03 2020

@author: Yashank Singh
"""

import math
import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import h5py
from sklearn.datasets.samples_generator import make_circles


train_data = pd.read_csv("C:\\Users\\Yashank Singh\\Desktop\\assn2_ELL409\\iris.csv")
train_set = train_data.values
X=train_set[:,0:-1]
Y=train_set[:,-1]
plt.figure()
plt.scatter(X[Y=='Iris-virginica',0],X[Y=='Iris-virginica',1],color='r', label ='Iris-virginica' )
plt.scatter(X[Y=='Iris-setosa',0],X[Y=='Iris-setosa',1],color='b', label ='Iris-setosa' )
plt.scatter(X[Y== 'Iris-versicolor',0],X[Y== 'Iris-versicolor',1],color='g', label = 'Iris-versicolor' )
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.legend()
plt.show()
plt.figure()
plt.scatter(X[Y=='Iris-virginica',2],X[Y=='Iris-virginica',3],color='r', label ='Iris-virginica' )
plt.scatter(X[Y=='Iris-setosa',2],X[Y=='Iris-setosa',3],color='b', label ='Iris-setosa' )
plt.scatter(X[Y== 'Iris-versicolor',2],X[Y== 'Iris-versicolor',3],color='g', label = 'Iris-versicolor' )
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.legend()
plt.show()

def calc_cost(X,W,centroids):
    cost =0
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            dist = np.sum((X[i,:]-centroids[j,:])**2)
            cost+=(W[i,j])*(dist**2)
    return cost 

def calc_centroids(X,W,p):
    #calculate centroids 
    centroids = np.divide(np.dot(X.T, W**p), np.sum(W, axis=0)).T
    return centroids        

def update_weights(X,centroids,p):
    W= np.zeros((X.shape[0],centroids.shape[0]))
    m = float(1/(p-1))
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            num= np.sum((X[i,:]-centroids[j,:].reshape(1,X.shape[1]))**2,axis=1)
            den=0
            for k in range(centroids.shape[0]):
                dist = np.sum((X[i,:]-centroids[k,:].reshape(1,X.shape[1]))**2,axis=1)
                den+=  math.pow((num/dist),m)
            W[i][j]=1/(float(den))
    return W        

def get_clusters(X,W,k):
    cluster_indices = np.argmax(W,axis=1) #getting the cluster indices as the ones with highest membership values
    clusters={}
    for i in range(k):
        clusters[i]=[]
    for i in range(X.shape[0]):
        clusters[cluster_indices[i]].append(i)
    return clusters

def predict(clusters,Y):
    cluster_labels=[]
    for i in range(len(clusters)):
        unique, count = np.unique(Y[clusters[i]],return_counts=True) #getting the majority vote as label in a cluster 
        if(len(unique)>0):
            cluster_labels.append(unique[np.argmax(count)]) #assigning lables to the points in a cluster
        else:
            cluster_labels.append(-1)
    return cluster_labels    

def accu(Y,clusters,labels):
    acc=0
    for i in range(len(clusters)):
        acc+=np.sum(Y[clusters[i]]==labels[i])
    return float(acc/Y.shape[0])    

def fuzzy_cmeans(X,k,p,num_iter):
    #initialize fuzzy matrix
    W=np.random.randint(low=1, high=100, size=(X.shape[0],k))
    #make sum of every row =1
    W= W/(np.sum(W,axis=0))
    for i in range(num_iter):
        #calculate centroids
        centroids = calc_centroids(X,W,p)
        cost = calc_cost(X,np.power(W,p),centroids)/X.shape[0]
        print(cost)
        W= update_weights(X,centroids,p)
        #print(np.sum(W[1,:].reshape(1,k),axis=1))
    clusters = get_clusters(X,W,k)#dictionary containing list of data point indices belonging to a cluster
    return clusters

clusters = fuzzy_cmeans(X,3,1.16,20)
labels=predict(clusters,Y)
print(labels)
print("accuracy =" +str(accu(Y,clusters,labels)))


plt.figure()
plt.scatter(X[clusters[0],2], X[clusters[0],3],color='r')
plt.scatter(X[clusters[1],2], X[clusters[1],3],color='g')
plt.scatter(X[clusters[2],2], X[clusters[2],3],color='b')
plt.show()
#clusters = fuzzy_cmeans(X,3,1.2,50) best  acc=0.9 
