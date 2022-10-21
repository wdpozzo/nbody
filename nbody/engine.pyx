cimport cython                        
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from nbody.body cimport body_t, _create_system, _find_mergers, _merge_bodies
from nbody.hamiltonian cimport _hamiltonian, _gradients
from cpython cimport array
#from __future__ import print_function

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
cdef _one_step_icn(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it):

    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt
    cdef body_t tmp_b
  
    

    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
        
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 


        
        
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


    
    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    for k in range(nbodies):
        start[k] = bodies[k] 

                            
                
                           
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
    
    #calculate the final forward coordinates
    for i in range(nbodies):
    
        for j in range(3):
            bodies[i].q[j] += dt2*g[i][3+j]
            bodies[i].p[j] -= dt2*g[i][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    _free(mid_point)


    #anti non-physical oscillation condition on each step
    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
             
            if (tmp.q[j] == 0):
                K[k].q[j] = 0
                
            if (tmp.p[j] == 0):
                K[k].p[j] = 0


            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j])
                #K[k].q[j] = dt2/(0.5*tmp.p[j])

            #if (K[k].q[j] > 1):
                #dt2 = 2*tmp.p[j]
                #tmp.p[j] = dt2*0.5 
                #bodies[k].p[j] = tmp.p[j] + start[k].p[j]

                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j])
                #K[k].p[j] = dt2/(0.5*tmp.q[j])  
            
            #if (K[k].p[j] > 1):
                #dt2 = 2*tmp.q[j]
                #tmp.q[j] = dt2*0.5 
                #bodies[k].q[j] = tmp.q[j] + start[k].q[j]


            if (K[k].q[j] > 0.5):
                #tmp.p[j] = np.sqrt(2*dt2)
                dt2 = 0.99*tmp.p[j]*0.5
                #bodies[k].p[j] = tmp.p[j] + start[k].p[j]
                
            if (K[k].p[j] > 0.5):
                #tmp.q[j] = np.sqrt(2*dt2)
                dt2 = 0.99*tmp.q[j]*0.5
                #bodies[k].q[j] = tmp.q[j] + start[k].q[j]


            D[k][j+3] = tmp.q[j]*tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j]*tmp.p[j] + dt2*dt2
                     
    _free(start)
    _free(K)
        
    for i in range(nbodies):
        free(g[i])
 
    free(g);
    
    cdef list dx = []
    cdef list dy = []
    cdef list dz = []

    cdef list dpx = []
    cdef list dpy = []
    cdef list dpz = []

    for k in range(nbodies): 
        dx.append(D[k][3])
        dy.append(D[k][4])
        dz.append(D[k][5])

        dpx.append(D[k][0])
        dpy.append(D[k][1])
        dpz.append(D[k][2])

    for i in range(nbodies):
        free(D[i])
 
    free(D);

    return (dx, dy, dz, dpx, dpy, dpz, dt2)


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



    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
        
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]



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


    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
            
            ''' 
            if (tmp.q[j] == 0):
                K[k].q[j] = 0
                
            if (tmp.p[j] == 0):
                K[k].p[j] = 0

            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j]*tmp.p[j])
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j]*tmp.q[j])

            if (K[k].q[j] > 0.5):
                dt2 = tmp.p[j]*tmp.p[j]*0.5

            if (K[k].p[j] > 0.5):
                dt2 = tmp.q[j]*tmp.q[j]*0.5

            '''

            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j]*tmp.p[j])

            if (K[k].q[j] > 1):
                dt2 = 0.5*tmp.p[j]*tmp.p[j]
                #tmp.p[j] = dt2*0.5 
                #bodies[k].p[j] = tmp.p[j] + start[k].p[j]
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j]*tmp.q[j])  
            
            if (K[k].p[j] > 1):
                dt2 = 0.5*tmp.q[j]*tmp.q[j]
                #tmp.q[j] = dt2*0.5 
                #bodies[k].q[j] = tmp.q[j] + start[k].q[j]
            
            #dq[k][j] = tmp.q[j]*tmp.q[j] + dt*dt
            #dp[k][j] = tmp.p[j]*tmp.p[j] + dt*dt
                                    
    _free(start)
    _free(K)

    
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



    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
        
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]



    _gradients(g, bodies, nbodies, order)
   
    for k in range(nbodies):
        mass = bodies[k].mass
                
        for j in range(3):
            bodies[k].q[j] += dt2*g[k][3+j]
            bodies[k].p[j] -= dt2*g[k][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    




    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
             
            if (tmp.q[j] == 0):
                K[k].q[j] = 0
   
            if (tmp.p[j] == 0):
                K[k].p[j] = 0

            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j]*tmp.p[j])
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j]*tmp.q[j])

            if (K[k].q[j] > 0.5):
                dt2 = tmp.p[j]*tmp.p[j]*0.5

            if (K[k].p[j] > 0.5):
                dt2 = tmp.q[j]*tmp.q[j]*0.5

            '''
            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(0.5*tmp.p[j])

            if (K[k].q[j] > 1):
                #dt2 = 2*tmp.p[j]
                tmp.p[j] = dt2*0.5 
                bodies[k].p[j] = tmp.p[j] + start[k].p[j]
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(0.5*tmp.q[j])  
            
            if (K[k].p[j] > 1):
                #dt2 = 2*tmp.q[j]
                tmp.q[j] = dt2*0.5 
                bodies[k].q[j] = tmp.q[j] + start[k].q[j]
            '''
            #dq[k][j] = tmp.q[j]*tmp.q[j] + dt*dt
            #dp[k][j] = tmp.p[j]*tmp.p[j] + dt*dt
                                    
    _free(start)
    _free(K)
    
    for i in range(nbodies):
        free(g[i])
 
    free(g);
    
    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _one_step_rk(body_t *bodies, unsigned int nbodies, long double dt, int order):

    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef np.ndarray[long double,mode="c",ndim=1] k1_q = np.zeros(3, dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=1] k1_p = np.zeros(3, dtype = np.longdouble) 
    cdef np.ndarray[long double,mode="c",ndim=1] k2_q = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k2_p = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k3_q = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k3_p = np.zeros(3, dtype = np.longdouble)   
    cdef np.ndarray[long double,mode="c",ndim=1] k4_q = np.zeros(3, dtype = np.longdouble)
    cdef np.ndarray[long double,mode="c",ndim=1] k4_p = np.zeros(3, dtype = np.longdouble) 
    cdef np.ndarray[long double,mode="c",ndim=1] k5_q = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k5_p = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k6_q = np.zeros(3, dtype = np.longdouble)    
    cdef np.ndarray[long double,mode="c",ndim=1] k6_p = np.zeros(3, dtype = np.longdouble)
               
    cdef body_t *tmp_q = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp_q == NULL:
        raise MemoryError

    cdef body_t *tmp_p = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp_p == NULL:
        raise MemoryError
        
    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))

    cdef long double **g_q = <long double **>malloc(nbodies*sizeof(long double *))    
    if g_q == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g_q[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g_q[i] == NULL:
            raise MemoryError
        memset(g_q[i], 0, 6*sizeof(long double))
        
    cdef long double **g_p = <long double **>malloc(nbodies*sizeof(long double *))    
    if g_p == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g_p[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g_p[i] == NULL:
            raise MemoryError
        memset(g_p[i], 0, 6*sizeof(long double))





    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
        
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]




               
    _gradients(g, bodies, nbodies, order)
   
    for k in range(nbodies):      
            
        mass = bodies[k].mass        
        tmp_q[k].mass = mass
        tmp_p[k].mass = mass               

        #k1   
        for j in range(3):
            
            k1_q[j] = dt2*g[k][3+j]
            k1_p[j] = -dt2*g[k][j] 
                     
            tmp_q[k].q[j] = bodies[k].q[j] + 0.5*k1_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + 0.5*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 0.5*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 0.5*k1_p[j]
                     
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)

    for k in range(nbodies): 
        #k2
        for j in range(3):
            
            k2_q[j] = dt2*g_q[k][3+j]   
            k2_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 0.5*k2_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + 0.5*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 0.5*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 0.5*k2_p[j]                 

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):                       
        #k3
        for j in range(3):
            
            k3_q[j] = dt2*g_q[k][3+j]   
            k3_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + k3_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + k3_p[j]                 

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k4
        for j in range(3):
            
            k4_q[j] = dt2*g_q[k][3+j]   
            k4_p[j] = -dt2*g_p[k][j]

            bodies[k].q[j] += (1./6.)*k1_q[j] + (1./3.)*k2_q[j] + (1./3.)*k3_q[j] + (1./6.)*k4_q[j]   
            bodies[k].p[j] += (1./6.)*k1_p[j] + (1./3.)*k2_p[j] + (1./3.)*k3_p[j] + (1./6.)*k4_p[j]

    '''        
        #k1   
        for j in range(3):
            
            k1_q[j] = dt2*g[k][3+j]
            k1_p[j] = -dt2*g[k][j] 
                     
            tmp_q[k].q[j] = bodies[k].q[j] + 0.25*k1_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + 0.25*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 0.25*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 0.25*k1_p[j]
                     
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)

    for k in range(nbodies): 
        #k2
        for j in range(3):
            
            k2_q[j] = dt2*g_q[k][3+j]   
            k2_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 3./32.*k1_q[j] + 9./32.*k2_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + 3./8.*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 3./8.*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 3./32.*k1_p[j] + 9./32.*k2_p[j]                 

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):                       
        #k3
        for j in range(3):
            
            k3_q[j] = dt2*g_q[k][3+j]   
            k3_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 1932./2197.*k1_q[j] - 7200./2197.*k2_q[j] +  7296./2197.*k3_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + 12./13.*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 12./13.*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 1932./2197.*k1_p[j] - 7200./2197.*k2_p[j] +  7296./2197.*k3_p[j]                 

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k4
        for j in range(3):
            
            k4_q[j] = dt2*g_q[k][3+j]   
            k4_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 439./216.*k1_q[j] - 8*k2_q[j] +  3680./513.*k3_q[j] - 845./4104.*k4_q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] + dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + dt2  
            tmp_p[k].p[j] = bodies[k].p[j] + 439./216.*k1_p[j] - 8*k2_p[j] +  3680./513.*k3_p[j] - 845./4104.*k4_p[j]                  

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k5
        for j in range(3):
            
            k5_q[j] = dt2*g_q[k][3+j]   
            k5_p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] - 8./27.*k1_q[j] + 2*k2_q[j] -  3544./2565.*k3_q[j] + 1859./4104.*k4_q[j] - 11./40.*k5_q[j]   
            tmp_q[k].p[j] = bodies[k].p[j] + 0.5*dt2     
                 
            tmp_p[k].q[j] = bodies[k].q[j] + 0.5*dt2  
            tmp_p[k].p[j] = bodies[k].p[j] - 8./27.*k1_p[j] + 2*k2_p[j] -  3544./2565.*k3_p[j] + 1859./4104.*k4_p[j] - 11./40.*k5_p[j]                     

            memset(g_q[k], 0, 6*sizeof(long double))
            memset(g_p[k], 0, 6*sizeof(long double))
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)    

    for k in range(nbodies):           
        #k6
        for j in range(3):
            
            k6_q[j] = dt2*g_q[k][3+j]   
            k6_p[j] = -dt2*g_p[k][j]         
            
            bodies[k].q[j] += (16./135.)*k1_q[j] + (6656./12825.)*k3_q[j] + (28561./56430.)*k4_q[j] - (9./50.)*k5_q[j] + (2./55.)*k6_q[j]            
            bodies[k].p[j] += (16./135.)*k1_p[j] + (6656./12825.)*k3_p[j] + (28561./56430.)*k4_p[j] - (9./50.)*k5_p[j] + (2./55.)*k6_p[j]       
    '''   
    _free(tmp_q)
    _free(tmp_p)




    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
             
            if (tmp.q[j] == 0):
                K[k].q[j] = 0
   
            if (tmp.p[j] == 0):
                K[k].p[j] = 0

            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j]*tmp.p[j])
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j]*tmp.q[j])

            if (K[k].q[j] > 0.5):
                dt2 = tmp.p[j]*tmp.p[j]*0.5

            if (K[k].p[j] > 0.5):
                dt2 = tmp.q[j]*tmp.q[j]*0.5
  
    _free(start)
    _free(K)
 

  

    for i in range(nbodies):
        free(g[i])
        free(g_p[i])
        free(g_q[i])
        
    free(g);
    free(g_p);
    free(g_q);
    
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
cpdef _H_2body(np.ndarray[long double, mode="c", ndim=1] mass,
          np.ndarray[long double, mode="c", ndim=1] x,
          np.ndarray[long double, mode="c", ndim=1] y,
          np.ndarray[long double, mode="c", ndim=1] z,
          np.ndarray[long double, mode="c", ndim=1] px,
          np.ndarray[long double, mode="c", ndim=1] py,
          np.ndarray[long double, mode="c", ndim=1] pz,
          np.ndarray[long double, mode="c", ndim=1] sx,
          np.ndarray[long double, mode="c", ndim=1] sy,
          np.ndarray[long double, mode="c", ndim=1] sz, 
          int order):
          
    cdef unsigned int n = 2
    cdef body_t *bodies = <body_t *> malloc(n * sizeof(body_t))
  
    _initialise(bodies, n, mass, x, y, z,
                px, py, pz, sx, sy, sz)
                
    #print(bodies[0], bodies[1])
                        
    h, t, v = _hamiltonian(bodies, n, order)
            
    
    return (h, t, v)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef run(long long int nsteps, long double dt, unsigned int order,
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
          unsigned int nthin,                                   
  #kernel-core-5.15.11-200.fc35.x86_64          
          unsigned int buffer_length, unsigned int plot_step):
    
    from tqdm import tqdm
    import pickle

    cdef unsigned int f_index
    cdef unsigned int n = len(mass)
    cdef unsigned int Neff = nsteps/(nthin*plot_step)
    #cdef body_t tmp

    '''    
    cdef long double ***D = <long double ***>malloc(Neff*sizeof(long double **))    
    if D == NULL:
        raise MemoryError
        
    for j in range(Neff):
        D[j] = <long double **>malloc(nbodies*sizeof(long double *)) #FIXME: for the spins
        if D[j] == NULL:
            raise MemoryError

        for k in range(6):
            D[j][k] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
            if D[j][k] == NULL:
                raise MemoryError
            memset(D[j][k], 0, 6*sizeof(long double))
    '''

    cdef list t_sim = []
    cdef list D = [[[0 for u in range(6)] for k in range(n)] for i in range(Neff)]

    cdef list dx = []
    cdef list dy = []
    cdef list dz = []
    cdef list dpx = []
    cdef list dpy = []
    cdef list dpz = []

    cdef long long int i

    cdef body_t *bodies = <body_t *> malloc(n * sizeof(body_t))
    cdef list solution = []
    cdef list H = []
    cdef list V = []
    cdef list T = []
    cdef long double h, t, v
    cdef long double time = 0.
    cdef long double dt_tmp = 0.

    #cdef long int nsteps = nsteps
    
    _initialise(bodies, n, mass, x, y, z,
                px, py, pz, sx, sy, sz)
                
    
    cdef unsigned int n_sol = 0
    
    #for i in range(nsteps):
    for i in tqdm(np.arange(nsteps)):
        '''
        #store the initial configuration 
        if (i == 0):
            solution.append([bodies[j] for j in range(n)])
            h, t, v = _hamiltonian(bodies, n, order)
        
            H.append(h)
            T.append(t)
            V.append(v)      

            t_sim.append(0.) 

            D[0][:][0] = 0. 
            D[0][:][1] = 0.  
            D[0][:][2] = 0. 
            D[0][:][3] = 0.  
            D[0][:][4] = 0.
            D[0][:][5] = 0.    
        '''

        # check for mergers
        n = _merge(bodies, n)
        
        # evolve forward in time
        #_one_step_eu(bodies, n, dt, order)
        #_one_step_lp(bodies, n, dt, order)
        #_one_step_rk(bodies, n, dt, order)
        dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_icn(bodies, n, dt, order, ICN_it)

        time += dt2_tmp    
        # store 1 every nthin steps 

        if ( (i+1)%nthin == 0.):    
            solution.append([bodies[j] for j in range(n)])
            h, t, v = _hamiltonian(bodies, n, order)
        
            H.append(h)
            T.append(t)
            V.append(v)
            
        # divide in files with buffer_lenght steps each    
        if ( (i+1)%buffer_length == 0.):
        
            pickle.dump(solution, open('solution_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(T, open('kinetic_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(V, open('potential_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(H, open('hamiltonian_{}_order{}.pkl'.format(n_sol, order),'wb'))
            n_sol += 1
            H        = []
            T        = []
            V        = []
            solution = []

        if ( (i+1)%(nthin*plot_step) == 0):

            f_index = (i+1)/(nthin*plot_step) - 1         
            t_sim.append(time)     
                

            for k in range(n):
                D[f_index][k][0] = dx[k] 
                D[f_index][k][1] = dy[k]  
                D[f_index][k][2] = dz[k] 
                D[f_index][k][3] = dpx[k]  
                D[f_index][k][4] = dpy[k]
                D[f_index][k][5] = dpz[k] 

    
    return (D, t_sim)
