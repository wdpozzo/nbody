import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def CM_system(long double [3]p1, long double [3]p2, long double [3]q1, long double [3]q2):
	
	cdef unsigned int i
	cdef long double [k][3]q_rel
	cdef long double [k][3]p_rel


	q_rel = np.array([q1[:] - q2[:]])
	p_rel = np.array([p1[:] - p2[:]])
	
	return (q_rel, p_rel)
