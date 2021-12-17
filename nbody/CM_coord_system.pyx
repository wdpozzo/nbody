import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def CM_system(list p1, list p2, list q1, list q2):

    cdef unsigned int i
    cdef np.ndarray[long double,mode="c",ndim=1] q_rel = np.zeros(3, dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=1] p_rel = np.zeros(3, dtype = np.longdouble)
    
    for i in range(3):
    
        q_rel[i] = np.array([q1[i] - q2[i]])       
        p_rel[i] = np.array([p1[i] - p2[i]])
    
    return (q_rel, p_rel)
    
