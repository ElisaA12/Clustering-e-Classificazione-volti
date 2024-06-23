import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm

def eigen(A):
    M, N = A.shape 
    L = np.dot(np.transpose(A),A)
    print(L.shape)
    valori, vettori = eig(L)
    
    ind =np.argsort(valori)[::-1]
    sort_eigenvalue = valori[ind]
    vettori = vettori[:, ind]
    U = np.dot(A,vettori)

    U= U/norm(U)
    
    return U,sort_eigenvalue
