import numpy as np
cimport numpy as np
cimport cython
#from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

def CM_system(np.ndarray[double, mode="c", ndim=2] p1, np.ndarray[double, mode="c", ndim=2] p2, np.ndarray[double, mode="c", ndim=2] q1, np.ndarray[double, mode="c", ndim=2] q2, long int Neff, double m1, double m2):

    cdef unsigned int i
    cdef np.ndarray[long double,mode="c",ndim=2] q_rel = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] p_rel = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] q_cm = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] p_cm = np.zeros((Neff,3), dtype = np.longdouble)
    #cdef np.ndarray[long double,mode="c",ndim=2] q = np.zeros((Neff,3), dtype = np.longdouble)
    #cdef np.ndarray[long double,mode="c",ndim=2] p = np.zeros((Neff,3), dtype = np.longdouble)
    
    
    for i in range(Neff):

        q_rel[i,:] = np.array([q1[i,:] - q2[i,:]])       
        p_rel[i,:] = np.array([(m2*p1[i,:] - m1*p2[i,:])/(m1 + m2)])
        
        q_cm[i,:] = np.array([(m1*q1[i,:] + m2*q2[i,:])/(m1+m2)])
        p_cm[i,:] = np.array([p1[i,:] + p2[i,:]])
           
    return (q_rel, p_rel, q_cm, p_cm)
