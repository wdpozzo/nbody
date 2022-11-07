import numpy as np
cimport numpy as np
from nbody.body cimport body_t
    
cdef _one_step_icn(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it) 

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

cdef void _one_step_lp(body_t *bodies, unsigned int nbodies, long double dt, int order) 

cdef void _one_step_eu(body_t *bodies, unsigned int nbodies, long double dt, int order)

cdef _one_step_rk(body_t *bodies, unsigned int nbodies, long double dt, int order)
