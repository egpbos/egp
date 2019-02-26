#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np

def cmultiply(np.ndarray[double, ndim=4] array, np.ndarray[double, ndim=4] result, double factor):
    cdef unsigned int i, j, k, l
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            for k in range(0, array.shape[2]):
                for l in range(0, array.shape[3]):
                    result[i,j,k,l] = factor*array[i,j,k,l]
