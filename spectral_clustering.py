import math
import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import h5py
from sklearn.datasets.samples_generator import make_circles


train_data = pd.read_csv("C:\\Users\\Yashank Singh\\Desktop\\assn2_ELL409\\iris.csv")
train_set = train_data.values
X_train=train_set[:,0:-1]
Y_train=train_set[:,-1]

data_sets = ['be3', 'happy', 'hm', 'sp', 'tar']
mat = h5py.File("C:\\Users\\Yashank Singh\\Desktop\\assn2_ELL409\\data.mat")


def epsilon_affinity_graph(X,eps):
    m= X.shape[0]
    W = np.zeros((m,m)) #edge weight graph
    D = np.zeros((m,m)) #Degree graph
    for i in range(m):
        for j in range(m):
            dist = np.linalg.norm(X[i,:]-X[j,:])
            #print(dist)
            if(dist<=eps and i!=j):
                W[i,j]=1
                D[i,i]+=1
    return W,D

def full_connected_graph(X,sigma,):
    m= X.shape[0]
    W = np.zeros((m,m)) #edge weight graph
    D = np.zeros((m,m)) #Degree graph
    for i in range(m):
        for j in range(m):
            norm = np.linalg.norm(X[i,:]-X[j,:])
            dist = norm*norm
            w=np.exp(- dist / (2 * sigma**2))
            if(i!=j):
                W[i,j]=w
                D[i,i]+=w
    return W,D

def knn_graph(X,k):
    m= X.shape[0]
    W = np.zeros((m,m)) #edge weight graph
    D = np.zeros((m,m)) #Degree graph
    knn={}
    for i in range(m):
        d={}
        for j in range(m):
            if(i!=j):
                dist = np.linalg.norm(X[i,:]-X[j,:])
                d[j]=dist               
        d=sorted(d.items(), key=lambda item: item[1])
        k_list=[]
        for l in range(k):
            k_list.append(d[l][0])
        knn[i]=k_list
#now we have the k nearest neighbour list for every sample
    for i in range(m):
        for j in range(m):
            if (j in knn[i] or i in knn[j]):
                W[i,j]=1
                D[i,i]+=1
    return W,D
#un normalized spectral clustering
def unnorm_sc(X, num_eig, graph_used, graph_param, num_iter):
    if(graph_used=="epsilon_affinity_graph"):
        W,D = epsilon_affinity_graph(X,graph_param)
    elif(graph_used=="knn_graph"):
        W,D=knn_graph(X,graph_param)
    else:
        W,D = full_connected_graph(X,graph_param)
#calculate graph laplacian
    L=D-W
    eig_vals, U = eigh(L, eigvals=(0,num_eig-1))
    clusters = k_means(U,num_eig, num_iter)
    return clusters
#print(unnorm_sc(X,5,"knn_graph",8).shape)
def k_means(U,k,num_iter):
    m=U.shape[0]
    n=U.shape[1]
    centroids = np.ndarray(shape=(k,n))
    for i in range(k):
        rand=rd.randint(0,m-1)
        centroids[i,:]=U[rand,:]
        dists = np.ndarray(shape=(m,k))
    for iter in range(num_iter):
        for i in range(k):
            dists[:,i] = np.sum((U-centroids[i,:])**2,axis =1)
        min_dist_centroid= np.argmin(dists, axis=1)
        cluster = {}
        for i in range(k):
            cluster[i]= []
        for i in range(m):
            cluster[min_dist_centroid[i]].append(U[i])
        for i in range(k):
            centroids[i,:] = np.mean(cluster[i],axis=0)
    clusters={}
    for i in range(k):
        clusters[i]=[]
    for i in range(m):
        clusters[min_dist_centroid[i]].append(i)
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

#X, clusters = make_circles(n_samples=1000, noise=.08, factor=.5, random_state=0)
#plt.scatter(X[:,0], X[:,1])

clusters = unnorm_sc(X_train,3,"epsilon_affinity_graph",3.8, 50)
labels=predict(clusters,Y_train)
print("accurcy =" +str(accu(Y_train,clusters,labels)))
plt.figure()
plt.scatter(X_train[clusters[0],2], X_train[clusters[0],3])
plt.scatter(X_train[clusters[1],2], X_train[clusters[1],3])
plt.scatter(X_train[clusters[2],2], X_train[clusters[2],3])
plt.show()
clusters = k_means(X_train,3,20)
labels=predict(clusters,Y_train)
print("accurcy =" +str(accu(Y_train,clusters,labels)))
plt.figure()
plt.scatter(X_train[clusters[0],2], X_train[clusters[0],3])
plt.scatter(X_train[clusters[1],2], X_train[clusters[1],3])
plt.scatter(X_train[clusters[2],2], X_train[clusters[2],3])
plt.show()

data_set_params = [.01, .01, .05, 1, .07]
X=np.array(mat['DB/' + data_sets[3]]).transpose()
clusters = unnorm_sc(X,3,"full_connected_graph", data_set_params[3], 50)
plt.figure()
plt.scatter(X[clusters[0],0], X[clusters[0],1])
plt.scatter(X[clusters[1],0], X[clusters[1],1])
plt.scatter(X[clusters[2],0], X[clusters[2],1])
plt.show()
clusters = k_means(X,3,20)
plt.figure()
plt.scatter(X[clusters[0],0], X[clusters[0],1])
plt.scatter(X[clusters[1],0], X[clusters[1],1])
plt.scatter(X[clusters[2],0], X[clusters[2],1])
plt.show()

#clusters = unnorm_sc(X,2,"full_connected_graph", 0.05, 100) concentric circles
#clusters = unnorm_sc(X,3,"full_connected_graph", 0.01, 100) dataset0,1
#clusters = unnorm_sc(X,2,"full_connected_graph", 0.05, 100)dataset2
#clusters = unnorm_sc(X,3,"full_connected_graph", 1, 100)dataset3
#clusters = unnorm_sc(X,3,"full_connected_graph",.07, 100)dataset4
