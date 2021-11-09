cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from nbody.body cimport body_t, _create_system, _find_mergers, _merge_bodies
from nbody.hamiltonian cimport _hamiltonian, _gradients

cdef unsigned int _merge(body_t *bodies, unsigned int nbodies):
    cdef int i_remove = -1
    cdef int i_survive = -1
    # first of all check for mergers (assume we get at most 1 per time step)
    i_survive, i_remove = _find_mergers(bodies, nbodies)

    if i_survive != -1 and i_remove != -1:
        _merge_bodies(bodies, i_survive, i_remove, nbodies)
        nbodies -= 1
    return nbodies

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _one_step(body_t *bodies, unsigned int nbodies, long double dt, int order):

    cdef unsigned int i,j,k
    cdef long double dtsquare = dt*dt
    cdef body_t b
    cdef body_t tmp_b
    cdef body_t *mid_point = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point == NULL:
        raise MemoryError
        
    cdef body_t *mid_point_2 = <body_t *>malloc(nbodies*sizeof(body_t))

    if mid_point_2 == NULL:
        raise MemoryError

    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))
    if g == NULL:
        raise MemoryError
    
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double))#FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))
        

    _gradients(g, bodies, nbodies, order)

    # FIXME: spins are not evolving!
    # iteration 0
    for i in range(nbodies):
        mass = bodies[i].mass
        mid_point[i].mass = mass
        
        for j in range(3):

            tmp_b.q[j] = bodies[i].q[j] + dtsquare*g[i][3+j]
            mid_point[i].q[j] = 0.5*(tmp_b.q[j] + bodies[i].q[j])

            tmp_b.p[j] = bodies[i].p[j] - dtsquare*g[i][j]
            mid_point[i].p[j] = 0.5*(tmp_b.p[j] + bodies[i].p[j])

            tmp_b.s[j] = bodies[i].s[j]
            bodies[i].s[j] = 0.5*(tmp_b.s[j] + bodies[i].s[j])

    # update the gradient
    for i in range(nbodies):
        memset(g[i], 0, 6*sizeof(long double))
    _gradients(g, mid_point, nbodies, order)
    # iteration 1
    for i in range(nbodies):
        
        mass = bodies[i].mass
        mid_point_2[i].mass = mass
        
        for j in range(3):

            tmp_b.q[j] = bodies[i].q[j] + dtsquare*g[i][3+j]
            mid_point_2[i].q[j] = 0.5*(tmp_b.q[j] + bodies[i].q[j])

            tmp_b.p[j] = bodies[i].p[j] - dtsquare*g[i][j]
            mid_point_2[i].p[j] = 0.5*(tmp_b.p[j] + bodies[i].p[j])

            tmp_b.s[j] = bodies[i].s[j]
            mid_point_2[i].s[j] = 0.5*(tmp_b.s[j] + bodies[i].s[j])

    
    # update the gradient
    for i in range(nbodies):
        memset(g[i], 0, 6*sizeof(long double))
    _gradients(g, mid_point_2, nbodies, order)

    for i in range(nbodies):
        mass = bodies[i].mass
        for j in range(3):
            bodies[i].q[j] += dtsquare*g[i][3+j]
            bodies[i].p[j] -= dtsquare*g[i][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    _free(mid_point)
    _free(mid_point_2)
    for i in range(nbodies):
        free(g[i])
 
    free(g);
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
                      long double[:] sz):

    _create_system(bodies, n, mass, x, y, z, px, py, pz, sx, sy, sz)
    return 

cdef void _free(body_t *s) nogil:
    free(<void *>s)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def run(unsigned int nsteps, long double dt, int order,
          np.ndarray[long double, mode="c", ndim=1] mass,
          np.ndarray[long double, mode="c", ndim=1] x,
          np.ndarray[long double, mode="c", ndim=1] y,
          np.ndarray[long double, mode="c", ndim=1] z,
          np.ndarray[long double, mode="c", ndim=1] px,
          np.ndarray[long double, mode="c", ndim=1] py,
          np.ndarray[long double, mode="c", ndim=1] pz,
          np.ndarray[long double, mode="c", ndim=1] sx,
          np.ndarray[long double, mode="c", ndim=1] sy,
          np.ndarray[long double, mode="c", ndim=1] sz):
    
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
        # check for mergers
        n = _merge(bodies, n)
        # evolve forward in time
        _one_step(bodies, n, dt, order)
        # store 1 every 10 steps
        if i%10 == 0:
            solution.append([bodies[i] for i in range(n)])
            H.append(_hamiltonian(bodies, n, order))
                     
    return solution,H
