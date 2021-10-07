import numpy as np
cimport numpy as np
from nbody.body cimport body_t
    
cdef void _one_step(body_t *bodies, unsigned int nbodies, long double dt, int order)

cdef void _initialise(body_t *bodies,
                      unsigned int n,
                      long double[:] mass,
                      long double[:] x,
                      long double[:] y,
                      long double[:] z,
                      long double[:] px,
                      long double[:] py,
                      long double[:] pz,
                      long double[:] sx,
                      long double[:] sy,
                      long double[:] sz)

cdef void _free(body_t *s) nogil
