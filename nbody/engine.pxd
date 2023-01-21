import numpy as np
cimport numpy as np
from nbody.body cimport body_t
    
cdef _one_step_icn(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it) 

cdef _one_step_hermite(body_t *bodies, unsigned int nbodies, long double dt, unsigned int int_order, unsigned int order)

cdef estimate_error_bs(long double q_4th, long double p_4th, long double q_2nd, long double p_2nd, double tol, unsigned int order) 

cdef acc(np.ndarray[long double, mode="c", ndim=1] q, long double mass, long double r, int nbodies) 

cdef jerk(np.ndarray[long double, mode="c", ndim=1] q, np.ndarray[long double, mode="c", ndim=1] p, long double mass, long double r, int nbodies) 

cdef _one_step_FR(body_t *bodies, unsigned int nbodies, long double dt, unsigned int order)

cdef _one_step_icn_mod(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it) 

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

cdef _one_step_sv(body_t *bodies, unsigned int nbodies, long double dt, int order) 

cdef _one_step_lp(body_t *bodies, unsigned int nbodies, long double dt, int order)

cdef _one_step_gar(body_t *bodies, unsigned int nbodies, long double dt, int order) 

cdef _one_step_lw(body_t *bodies, unsigned int nbodies, long double dt, int order) 

cdef _one_step_eu(body_t *bodies, unsigned int nbodies, long double dt, int order)

cdef _one_step_rk(body_t *bodies, unsigned int nbodies, long double dt, int order)
