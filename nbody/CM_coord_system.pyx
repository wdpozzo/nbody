import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

def CM_system(np.ndarray[double, mode="c", ndim=2] p1, np.ndarray[double, mode="c", ndim=2] p2, np.ndarray[double, mode="c", ndim=2] q1, np.ndarray[double, mode="c", ndim=2] q2, long int Neff):

    cdef unsigned int i
    cdef np.ndarray[long double,mode="c",ndim=2] q_rel = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] p_rel = np.zeros((Neff,3), dtype = np.longdouble)
    
    for i in range(Neff):
    
        q_rel[i,:] = np.array([q1[i,:] - q2[i,:]])       
        p_rel[i,:] = np.array([p1[i,:] - p2[i,:]])
    
    return (q_rel, p_rel)
    
