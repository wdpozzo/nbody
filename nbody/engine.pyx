cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from nbody.body cimport body_t, _create_system
from nbody.hamiltonian cimport _hamiltonian, _gradients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _one_step(body_t *bodies, unsigned int nbodies, double dt, int order):

    cdef unsigned int i,j,k
    cdef double dtsquare = dt*dt
    cdef body_t b
    cdef body_t tmp_b
    cdef body_t *mid_point = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point == NULL:
        raise MemoryError
        
    cdef body_t *mid_point_2 = <body_t *>malloc(nbodies*sizeof(body_t))

    if mid_point_2 == NULL:
        raise MemoryError

    cdef double[:,:] g = _gradients(bodies, nbodies, order)

    # FIXME: spins are not evolving!
    # iteration 0
    for i in range(nbodies):
        mass = bodies[i].mass
        mid_point[i].mass = mass
        
        for j in range(3):

            tmp_b.q[j] = bodies[i].q[j] + dtsquare*g[i,3+j]
            mid_point[i].q[j] = 0.5*(tmp_b.q[j] + bodies[i].q[j])

            tmp_b.p[j] = bodies[i].p[j] + dtsquare*g[i,j]
            mid_point[i].p[j] = 0.5*(tmp_b.p[j] + bodies[i].p[j])

            tmp_b.s[j] = bodies[i].s[j]
            bodies[i].s[j] = 0.5*(tmp_b.s[j] + bodies[i].s[j])

    # update the gradient
    g = _gradients(mid_point, nbodies, order)
    # iteration 1
    for i in range(nbodies):
        
        mass = bodies[i].mass
        mid_point_2[i].mass = mass
        
        for j in range(3):

            tmp_b.q[j] = bodies[i].q[j] + dtsquare*g[i,3+j]
            mid_point_2[i].q[j] = 0.5*(tmp_b.q[j] + bodies[i].q[j])

            tmp_b.p[j] = bodies[i].p[j] + dtsquare*g[i,j]
            mid_point_2[i].p[j] = 0.5*(tmp_b.p[j] + bodies[i].p[j])

            tmp_b.s[j] = bodies[i].s[j]
            mid_point_2[i].s[j] = 0.5*(tmp_b.s[j] + bodies[i].s[j])

    
    # update the gradient
    g = _gradients(mid_point_2, nbodies, order)

    for i in range(nbodies):
        mass = bodies[i].mass
        for j in range(3):
            bodies[i].q[j] += dtsquare*g[i,3+j]
            bodies[i].p[j] += dtsquare*g[i,j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution

    _free(mid_point)
    _free(mid_point_2)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
                      double[:] sz):

    _create_system(bodies, n, mass, x, y, z, px, py, pz, sx, sy, sz)
    return 

cdef void _free(body_t *s):
    free(<void *>s)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def run(unsigned int nsteps, double dt, int order,
          np.ndarray[double, mode="c", ndim=1] mass,
          np.ndarray[double, mode="c", ndim=1] x,
          np.ndarray[double, mode="c", ndim=1] y,
          np.ndarray[double, mode="c", ndim=1] z,
          np.ndarray[double, mode="c", ndim=1] px,
          np.ndarray[double, mode="c", ndim=1] py,
          np.ndarray[double, mode="c", ndim=1] pz,
          np.ndarray[double, mode="c", ndim=1] sx,
          np.ndarray[double, mode="c", ndim=1] sy,
          np.ndarray[double, mode="c", ndim=1] sz):
    
    from tqdm import tqdm
    cdef unsigned int i,j
    cdef unsigned int n = len(mass)
    cdef body_t *bodies = <body_t *> malloc(n * sizeof(body_t))
    cdef list solution = []
    cdef list H = []
    
    
    _initialise(bodies, n, mass, x, y, z,
                px, py, pz, sx, sy, sz)
    solution.append([bodies[i] for i in range(n)])
    H.append(_hamiltonian(bodies, n, order))
    
    for i in tqdm(range(1,nsteps)):
        _one_step(bodies, n, dt, order)
        solution.append([bodies[i] for i in range(n)])
        H.append(_hamiltonian(bodies, n, order))
    
    return solution,H
