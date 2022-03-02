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
cdef void _one_step_icn(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it):

    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt
    cdef body_t tmp_b
    
 
    cdef body_t *mid_point = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point == NULL:
        raise MemoryError
        
    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))
        

    _gradients(g, bodies, nbodies, order)
           
    for i in range(ICN_it):   
        # FIXME: spins are not evolving!
        
        for k in range(nbodies):
            mass = bodies[k].mass
            mid_point[k].mass = mass 
        
            for j in range(3):
                
                tmp_b.q[j] = bodies[k].q[j] + dt2*g[k][3+j]
                mid_point[k].q[j] = 0.5*(tmp_b.q[j] + bodies[k].q[j])

                tmp_b.p[j] = bodies[k].p[j] - dt2*g[k][j]
                mid_point[k].p[j] = 0.5*(tmp_b.p[j] + bodies[k].p[j])

                tmp_b.s[j] = bodies[k].s[j]
                mid_point[k].s[j] = 0.5*(tmp_b.s[j] + bodies[k].s[j])

        # update the gradient
        for k in range(nbodies):
            memset(g[k], 0, 6*sizeof(long double))
        _gradients(g, mid_point, nbodies, order)
        
        #print(g[0][0])
        
    #calculate the final forward coordinates
    for i in range(nbodies):
        mass = bodies[i].mass
        
        for j in range(3):
            bodies[i].q[j] += dt2*g[i][3+j]
            bodies[i].p[j] -= dt2*g[i][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    _free(mid_point)
    
    for i in range(nbodies):
        free(g[i])
 
    free(g);
    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _one_step_lp(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef body_t *tmp_b = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp_b == NULL:
        raise MemoryError
        
    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))    

    _gradients(g, bodies, nbodies, order)

    for k in range(nbodies):
        mass = bodies[k].mass
        tmp_b[k].mass = mass         
        
        for j in range(3):
                
            tmp_b[k].q[j] = bodies[k].q[j] + dt2*g[k][3+j]

            tmp_b[k].p[j] = bodies[k].p[j] - dt2*g[k][j]

            tmp_b[k].s[j] = bodies[k].s[j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp_b, nbodies, order)       
    
    for k in range(nbodies):
        mass = tmp_b[k].mass
                
        for j in range(3):
            bodies[k].q[j] = tmp_b[k].q[j] + dt2*g[k][3+j]
            bodies[k].p[j] = tmp_b[k].p[j] - dt2*g[k][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    
    _free(tmp_b)
    
    for i in range(nbodies):
        free(g[i])
 
    free(g);
    
    return
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _one_step_eu(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt
        
    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))    

    _gradients(g, bodies, nbodies, order)
   
    for k in range(nbodies):
        mass = bodies[k].mass
                
        for j in range(3):
            bodies[k].q[j] += dt2*g[k][3+j]
            bodies[k].p[j] -= dt2*g[k][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    
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
def run(long long int nsteps, long double dt, int order,
          np.ndarray[long double, mode="c", ndim=1] mass,
          np.ndarray[long double, mode="c", ndim=1] x,
          np.ndarray[long double, mode="c", ndim=1] y,
          np.ndarray[long double, mode="c", ndim=1] z,
          np.ndarray[long double, mode="c", ndim=1] px,
          np.ndarray[long double, mode="c", ndim=1] py,
          np.ndarray[long double, mode="c", ndim=1] pz,
          np.ndarray[long double, mode="c", ndim=1] sx,
          np.ndarray[long double, mode="c", ndim=1] sy,
          np.ndarray[long double, mode="c", ndim=1] sz,
          unsigned int ICN_it,
          int nthin,                                   
  #kernel-core-5.15.11-200.fc35.x86_64          
          unsigned int buffer_length):
    
    from tqdm import tqdm
    import pickle
    cdef long long int i
    cdef int n = len(mass)
    cdef body_t *bodies = <body_t *> malloc(n * sizeof(body_t))
    cdef list solution = []
    cdef list H = []
    cdef list V = []
    cdef list T = []
    cdef long double h, t, v
    #cdef long int nsteps = nsteps
    
    _initialise(bodies, n, mass, x, y, z,
                px, py, pz, sx, sy, sz)
                
    #solution.append([bodies[i] for i in range(n)])
    #h, t, v = _hamiltonian(bodies, n, order)
    #H.append(h)
    #T.append(t)
    #V.append(v)
    
    cdef long int n_sol = 0
    
    #for i in range(nsteps):
    for i in tqdm(np.arange(nsteps)):
        # check for mergers
        n = _merge(bodies, n)
        # evolve forward in time
        #_one_step_eu(bodies, n, dt, order)
        _one_step_icn(bodies, n, dt, order, ICN_it)
        
        # store 1 every nthin steps        
        if (i+1)%nthin == 0:
        
            solution.append([bodies[i] for i in range(n)])
            h, t, v = _hamiltonian(bodies, n, order)
        
            H.append(h)
            T.append(t)
            V.append(v)
            
        # divide in files with buffer_lenght steps each    
        if (i+1)%buffer_length == 0:
        
            pickle.dump(solution, open('solution_{}.pkl'.format(n_sol),'wb'))
            pickle.dump(T, open('kinetic_{}.pkl'.format(n_sol),'wb'))
            pickle.dump(V, open('potential_{}.pkl'.format(n_sol),'wb'))
            pickle.dump(H, open('hamiltonian_{}.pkl'.format(n_sol),'wb'))
            n_sol += 1
            H        = []
            T        = []
            V        = []
            solution = []
            
    return 1
