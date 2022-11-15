import numpy as np
cimport numpy as np
cimport cython
import math
#from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef CM_system(np.ndarray[long double, mode="c", ndim=2] p1, np.ndarray[long double, mode="c", ndim=2] p2, np.ndarray[long double, mode="c", ndim=2] q1, np.ndarray[long double, mode="c", ndim=2] q2, long int Neff, double m1, double m2):

    cdef unsigned int i
    cdef np.ndarray[long double,mode="c",ndim=2] q_1 = q1
    cdef np.ndarray[long double,mode="c",ndim=2] p_1 = p1
    cdef np.ndarray[long double,mode="c",ndim=2] q_2 = q2
    cdef np.ndarray[long double,mode="c",ndim=2] p_2 = p2

    cdef np.ndarray[long double,mode="c",ndim=2] q_rel = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] p_rel = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] q_cm = np.zeros((Neff,3), dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=2] p_cm = np.zeros((Neff,3), dtype = np.longdouble)

    for i in range(Neff):

        q_rel[i,:] = np.array([q_1[i,:] - q_2[i,:]])       
        p_rel[i,:] = np.array([(m2*p_1[i,:] - m1*p_2[i,:])/(m1 + m2)])
        
        q_cm[i,:] = np.array([(m1*q_1[i,:] + m2*q_2[i,:])/(m1 + m2)])
        p_cm[i,:] = np.array([p_1[i,:] + p_2[i,:]])

    return (q_rel, p_rel, q_cm, p_cm)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef SpherToCart(np.ndarray[double, mode="c", ndim=1] RA, np.ndarray[double, mode="c", ndim=1] Decl, long double r):

    cdef unsigned int i
    cdef x = np.zeros(len(RA))
    cdef y = np.zeros(len(RA))
    cdef z = np.zeros(len(RA))
    
    for i in range(len(RA)):

        x[i] = r*math.sin(Decl[i])*math.cos(RA[i])
        y[i] = r*math.sin(Decl[i])*math.sin(RA[i])
        z[i] = r*math.cos(Decl[i])

    return (x, y, z)

cpdef CartToSpher(np.ndarray[long double, mode="c", ndim=1] x, np.ndarray[long double, mode="c", ndim=1] y, np.ndarray[long double, mode="c", ndim=1] z):

    cdef unsigned int i
    cdef r = np.zeros(len(x))
    cdef theta = np.zeros(len(x))
    cdef phi = np.zeros(len(x))
    
    for i in range(len(x)):

        r[i] = math.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
        theta[i] = math.atan2(y[i], x[i])
        phi[i] = math.atan2(math.sqrt(x[i]*x[i] + y[i]*y[i]), z[i])

    return (r, theta, phi)

'''
cpdef CartToSpher(np.ndarray[double, mode="c", ndim=1] x, np.ndarray[double, mode="c", ndim=1] y, np.ndarray[double, mode="c", ndim=1] z):

    cdef unsigned int i
    cdef RA = np.zeros(len(RA))
    cdef Decl = np.zeros(len(RA))
    
    for i in range(len(RA)):

        x[i] = r*math.sin(Decl[i])*math.cos(RA[i])
        y[i] = r*math.sin(Decl[i])*math.sin(RA[i])
        z[i] = r*math.cos(Decl[i])

    return (RA, Decl)
'''
