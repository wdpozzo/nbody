import numpy as np
cimport numpy as np
from nbody.body cimport body_t, system_t
    
cdef void _one_step(body_t *bodies, unsigned int nbodies, double dt, int order)

cdef void _initialise(body_t *bodies,
                      unsigned int n,
                      double[:] mass,
                      double[:] x,
                      double[:] y,
                      double[:] z,
                      double[:] px,
                      double[:] py,
                      double[:] pz,
                      double[:] sx,
                      double[:] sy,
                      double[:] sz)

cdef void _free(body_t *s)
