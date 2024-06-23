from collections import Counter
from math import dist
import numpy as np
import collections

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1-point2, axis=0)

def distances_euc(x,y):
    return np.sqrt(np.sum(np.square(x-y),0))

def get_distance(element):
    return element['distance']

def knn_predict(X_train, X_test, label_train, K, k):

    lbl_test = []
    i = 0
        
    dist=[]
    class_k=[]
    for j in range(X_train.shape[0]):
        train=X_train[j:j+1,:]
        distance = list(euclidean_distance(X_test, train))
        #print(distance)
        elem ={'distance':distance, 'class':label_train[j]}
        dist.append(elem)
    
    dist.sort(key=get_distance)
    dist_k=dist[:k]
    for d in dist_k:
        class_k.append(d['class'])

    counter = Counter(class_k)

    prediction = counter.most_common()[0][0]
    lbl_test.append(prediction)

    return prediction


def knn_init(X_train, X_test, label,K, kn):
    y_test=0

    y_test = knn_predict(X_train, X_test, label, K, k=kn)

    return y_test
