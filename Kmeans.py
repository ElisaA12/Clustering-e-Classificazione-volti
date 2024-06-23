import numpy as np

# Funzione che mi permette di ritornare gli indici associati ad un valore
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def initCentroids(X,K):
    # X: N x D
    # random_idx: 1 x N 
    # centroids: K x D
    centroids= np.zeros((K,X.shape[1]))
    random_idx= np.array(np.random.permutation(X.shape[0]))
    centroids= X[random_idx[:K],:] #prendo i primi K centri
    return centroids

def km(X,centroids):
    # cluster: dizionario che ha come keys le classi e come items le immagini che appartengono a quella classe

    N,D = X.shape
    K= centroids.shape[0] 
    index= np.zeros(N) 
    cluster={el:[] for el in list(range(K))}
    i=0
    for i in range(N):
        k=0
        X_i = X[i:i+1,:]
        center_i = centroids[0:1,:]
        min_dist=np.linalg.norm( X_i - center_i ) #calcolo la distanza dal primo centroide e poi la confronto con gli altri centroidi
        for j in range(1,K):
            dist=np.linalg.norm(X[i,:] - centroids[j,:])
            if dist < min_dist:
                min_dist=dist
                k=j   
        index[i]=k
        cluster[k].append(X_i)
        
        
    #ricalcolimo i centri considerando gli elementi che sono stati assegnati alle classi
    centroids= np.zeros((K,D))
    for i in range(K):
        centroids[i,:] =  np.mean(cluster[i],0)
        
    return centroids, index


def kmeans_init(X,K,max_iterations):
  # X: N x D
  # K: numero di classi
  # centroids: K x D
  # confmat: K x K
  # index: 1 x N
  centroids=0
  index=0
  confmat=0

  #while(accuracy < 0.5):
  # Inizializzo randomincamente i centroidi
  centroids = initCentroids(X, K)
  for i in range(max_iterations):
    centr= centroids.copy()
    centroids, index= km(X, centroids)
    if(np.allclose(centroids, centr, rtol=1e-02, atol=1e-04, equal_nan=False)):
        break
  
  return centroids, index, confmat