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
    cdef list dt2_p_list = []
    cdef list dt2_q_list = []
    cdef np.ndarray[long double, mode="c", ndim=1] dt2_p_array = np.zeros((nbodies), dtype = np.longdouble) 
    cdef np.ndarray[long double, mode="c", ndim=1] dt2_q_array = np.zeros((nbodies), dtype = np.longdouble) 

    cdef body_t tmp_1
    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
   
    #cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    #if K == NULL:
    #    raise MemoryError       
        
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
                
                tmp_1.q[j] = bodies[k].q[j] + dt2*g[k][3+j]
                mid_point[k].q[j] = 0.5*(tmp_1.q[j] + bodies[k].q[j])

                tmp_1.p[j] = bodies[k].p[j] - dt2*g[k][j]
                mid_point[k].p[j] = 0.5*(tmp_1.p[j] + bodies[k].p[j])

                tmp_1.s[j] = bodies[k].s[j]
                mid_point[k].s[j] = 0.5*(tmp_1.s[j] + bodies[k].s[j])
        
        # update the gradient
        for k in range(nbodies):
            memset(g[k], 0, 6*sizeof(long double))
            
        _gradients(g, mid_point, nbodies, order)

    
    #calculate the final forward coordinates
    for i in range(nbodies):
    
        for j in range(3):
            bodies[i].q[j] += dt2*g[i][3+j]
            bodies[i].p[j] -= dt2*g[i][j]
            #bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution

    #anti non-physical oscillation condition on each step
    '''
    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
            
            if (tmp.p[j] == 0):
                K[k].q[j] = 0
                
            if (tmp.q[j] == 0):
                K[k].p[j] = 0

            if (tmp.p[j] != 0):
                K[k].q[j] = dt2/(tmp.p[j])
                #K[k].q[j] = dt2/(0.5*tmp.p[j])

            if (tmp.q[j] != 0):
                K[k].p[j] = dt2/(tmp.q[j])
                #K[k].p[j] = dt2/(0.5*tmp.q[j])  


                #if (K[k].q[j] > 1):
                    #dt2 = 2*tmp.p[j]
                    #tmp.p[j] = dt2*0.5 
                    #bodies[k].p[j] = tmp.p[j] + start[k].p[j]
            
                #if (K[k].p[j] > 1):
                    #dt2 = 2*tmp.q[j]
                    #tmp.q[j] = dt2*0.5 
                    #bodies[k].q[j] = tmp.q[j] + start[k].q[j]


            if (K[k].q[j] > 0.5):
                    #tmp.p[j] = np.sqrt(2*dt2)
                dt2_p_list.append(0.5*tmp.p[j])
                    #bodies[k].p[j] = tmp.p[j] + start[k].p[j]
                
            if (K[k].p[j] > 0.5):
                    #tmp.q[j] = np.sqrt(2*dt2)
                dt2_q_list.append(0.5*tmp.q[j])
                    #bodies[k].q[j] = tmp.q[j] + start[k].q[j]

            if (K[k].q[j] <= 0.5):
                    #tmp.p[j] = np.sqrt(2*dt2)
                dt2_p_list.append(dt2)
                    #bodies[k].p[j] = tmp.p[j] + start[k].p[j]
                
            if (K[k].p[j] <= 0.5):
                    #tmp.q[j] = np.sqrt(2*dt2)
                dt2_q_list.append(dt2)
                    #bodies[k].q[j] = tmp.q[j] + stacrt[k].q[j]
        
        dt2_q_array[k] = np.linalg.norm(dt2_q_list)
        dt2_p_array[k] = np.linalg.norm(dt2_p_list)
        
        dt2_q_list = []
        dt2_p_list = []

    dt2_min_p = min(dt2_p_array)
    dt2_min_q = min(dt2_q_array)

    #print(dt2_min_q, dt2_min_p, dt2)
    _free(K)

    if (min(dt2_min_p, dt2_min_q) < dt2):
        dt2 = min(dt2_min_p, dt2_min_q)

        # update the gradient
        for k in range(nbodies):
            memset(g[k], 0, 6*sizeof(long double))
        _gradients(g, start, nbodies, order)

        for i in range(ICN_it):   
        # FIXME: spins are not evolving!
        
            for k in range(nbodies):
                mass = bodies[k].mass
                mid_point[k].mass = mass 
        
                for j in range(3):
                
                    tmp_1.q[j] = start[k].q[j] + dt2*g[k][3+j]
                    mid_point[k].q[j] = 0.5*(tmp_1.q[j] + start[k].q[j])

                    tmp_1.p[j] = start[k].p[j] - dt2*g[k][j]
                    mid_point[k].p[j] = 0.5*(tmp_1.p[j] + start[k].p[j])
 
                    tmp_1.s[j] = start[k].s[j]
                    mid_point[k].s[j] = 0.5*(tmp_1.s[j] + start[k].s[j])
        
                # update the gradient
            for k in range(nbodies):
                memset(g[k], 0, 6*sizeof(long double))    
            _gradients(g, mid_point, nbodies, order)
    
            #calculate the final forward coordinates
        for k in range(nbodies):
            for j in range(3):
                bodies[k].q[j] = start[k].q[j] + dt2*g[k][3+j]
                bodies[k].p[j] = start[k].p[j] - dt2*g[k][j]
                #bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution  
    '''

    _free(mid_point)

    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]

            D[k][j+3] = tmp.q[j]*tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j]*tmp.p[j] + dt2*dt2
                  
    _free(start)
        
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
cdef _one_step_icn_mod(body_t *bodies, unsigned int nbodies, long double dt, int order, unsigned int ICN_it):

    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef body_t tmp_1
    cdef body_t tmp_2
  
    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
        
    cdef body_t *mid_point_1 = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point_1 == NULL:
        raise MemoryError

    cdef body_t *mid_point_2 = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point_2 == NULL:
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
            mid_point_1[k].mass = mass 
        
            for j in range(3):
                
                tmp_1.q[j] = bodies[k].q[j] + dt2*g[k][3+j]
                mid_point_1[k].q[j] = 0.5*(tmp_1.q[j] + bodies[k].q[j])

                tmp_1.p[j] = bodies[k].p[j] - dt2*g[k][j]
                mid_point_1[k].p[j] = 0.5*(tmp_1.p[j] + bodies[k].p[j])

                tmp_1.s[j] = bodies[k].s[j]
                mid_point_1[k].s[j] = 0.5*(tmp_1.s[j] + bodies[k].s[j])
        
        # update the gradient
        for k in range(nbodies):
            memset(g[k], 0, 6*sizeof(long double))
            
        _gradients(g, mid_point_1, nbodies, order)

        for k in range(nbodies):
            mass = bodies[k].mass
            mid_point_2[k].mass = mass 
        
            for j in range(3):
                
                tmp_2.q[j] = bodies[k].q[j] + dt2*g[k][3+j]
                mid_point_2[k].q[j] = 0.5*(tmp_2.q[j] + bodies[k].q[j])

                tmp_2.p[j] = bodies[k].p[j] - dt2*g[k][j]
                mid_point_2[k].p[j] = 0.5*(tmp_2.p[j] + bodies[k].p[j])

                tmp_2.s[j] = bodies[k].s[j]
                mid_point_2[k].s[j] = 0.5*(tmp_2.s[j] + bodies[k].s[j])        

        # update the gradient
        for k in range(nbodies):
            memset(g[k], 0, 6*sizeof(long double))
            
        _gradients(g, mid_point_2, nbodies, order)
   
    #calculate the final forward coordinates
    for i in range(nbodies):
    
        for j in range(3):
            bodies[i].q[j] += dt2*g[i][3+j]
            bodies[i].p[j] -= dt2*g[i][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution
    
    _free(mid_point_1)
    _free(mid_point_2)


    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]

            D[k][j+3] = tmp.q[j]*tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j]*tmp.p[j] + dt2*dt2
                     
    _free(start)
    #_free(K)
        
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
cdef _one_step_lp(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
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
    
    '''    
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]
    '''

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    _gradients(g, bodies, nbodies, order)


    for k in range(nbodies):
        start[k] = bodies[k] 

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
            '''

            D[k][j+3] = tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j] + dt2*dt2
                                    
    _free(start)
    #_free(K)
        
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
cdef _one_step_sv(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef body_t *tmp1 = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp1 == NULL:
        raise MemoryError
    #cdef body_t *tmp2 = <body_t *>malloc(nbodies*sizeof(body_t))
    #if tmp2 == NULL:
    #    raise MemoryError
        
    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))    

    cdef long double **g_temp = <long double **>malloc(nbodies*sizeof(long double *))    
    if g_temp == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g_temp[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g_temp[i] == NULL:
            raise MemoryError
        memset(g_temp[i], 0, 6*sizeof(long double))


    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
    
    '''    
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]
    '''

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    _gradients(g, bodies, nbodies, order)
    _gradients(g_temp, bodies, nbodies, order)

    for k in range(nbodies):
        start[k] = bodies[k] 

    for k in range(nbodies):            
        for j in range(3):
                
            tmp1[k].q[j] = bodies[k].q[j] + dt*g[k][j+3]
            tmp1[k].p[j] = bodies[k].p[j] - dt2*g[k][j]
            tmp1[k].s[j] = bodies[k].s[j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp1, nbodies, order) 

    for k in range(nbodies):            
        for j in range(3):

            tmp1[k].q[j] = bodies[k].q[j] + dt2*(g[k][j+3] + g_temp[k][j+3])
            tmp1[k].p[j] += - dt2*g[k][j]

    for k in range(nbodies):            
        for j in range(3):

            bodies[k].q[j] = tmp1[k].q[j]
            bodies[k].p[j] = tmp1[k].p[j]
            bodies[k].s[j] = tmp1[k].s[j]
    '''
    for k in range(nbodies):            
        for j in range(3):
                
            tmp1[k].q[j] = bodies[k].q[j] 
            tmp1[k].p[j] = bodies[k].p[j] - dt2*g[k][j]
            tmp1[k].s[j] = bodies[k].s[j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp1, nbodies, order)       

    for k in range(nbodies):  
        for j in range(3):

            tmp1[k].q[j] += dt2*g[k][3+j]

    for k in range(nbodies):            
        for j in range(3):

            tmp2[k].q[j] = tmp1[k].q[j] + dt2*g[k][3+j]
            tmp2[k].q[j] = tmp1[k].p[j] 
            tmp2[k].q[j] = tmp1[k].s[j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp2, nbodies, order)   

    for k in range(nbodies):            
        for j in range(3):

            bodies[k].q[j] = tmp2[k].q[j]
            bodies[k].p[j] = tmp2[k].p[j] - dt2*g[k][j]
            bodies[k].s[j] = tmp2[k].s[j]
    '''
    _free(tmp1)
    #_free(tmp2)

    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
            
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
            '''

            D[k][j+3] = tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j] + dt2*dt2
                                    
    _free(start)
    #_free(K)
        
    for i in range(nbodies):
        free(g[i])
 
    free(g);

    for i in range(nbodies):
        free(g_temp[i])
 
    free(g_temp);

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
cdef _one_step_gar(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef body_t *tmp1 = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp1 == NULL:
        raise MemoryError
    cdef body_t *tmp2 = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp2 == NULL:
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
    
    '''    
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]
    '''

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    _gradients(g, bodies, nbodies, order)


    for k in range(nbodies):
        start[k] = bodies[k] 


    for k in range(nbodies):            
        for j in range(3):
                
            tmp1[k].q[j] = bodies[k].q[j] + dt2*g[k][3+j]

            tmp1[k].p[j] = bodies[k].p[j]

            tmp1[k].s[j] = bodies[k].s[j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp1, nbodies, order)       

    for k in range(nbodies):            
        for j in range(3):

            tmp1[k].p[j] -= dt2*g[k][j]

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp1, nbodies, order)   
   
    for k in range(nbodies):
                
        for j in range(3):
            tmp2[k].q[j] = tmp1[k].q[j] 
            tmp2[k].p[j] = tmp1[k].p[j] - dt2*g[k][j]
            tmp2[k].s[j] = tmp1[k].s[j] 

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp2, nbodies, order)   

    for k in range(nbodies):
                
        for j in range(3):
            bodies[k].q[j] = tmp1[k].q[j] + dt2*g[k][j+3]
            bodies[k].p[j] = tmp1[k].p[j] 
            bodies[k].s[j] = tmp1[k].s[j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution  
    
    _free(tmp1)
    _free(tmp2)

    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
            
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
            '''

            D[k][j+3] = tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j] + dt2*dt2
                                    
    _free(start)
    #_free(K)
        
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


#'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _one_step_lw(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    cdef body_t tmp_b

    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))    


    cdef body_t *mid_point = <body_t *>malloc(nbodies*sizeof(body_t))
    if mid_point == NULL:
        raise MemoryError

    cdef body_t tmp

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError
    
    '''    
    cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    if K == NULL:
        raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]
    '''

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    _gradients(g, bodies, nbodies, order)


    for k in range(nbodies):
        start[k] = bodies[k] 


    for k in range(nbodies):
        mass = bodies[k].mass
        tmp_b.mass = mass         
        
        for j in range(3):
                
            tmp_b.q[j] = bodies[k].q[j] + dt*g[k][3+j]
            mid_point[k].q[j] = 0.5*(tmp_b.q[j] - bodies[k].q[j]) + dt2*g[k][3+j]
            
            tmp_b.p[j] = bodies[k].p[j] - dt*g[k][j]
            mid_point[k].p[j] = 0.5*(tmp_b.p[j] - bodies[k].p[j]) - dt2*g[k][j]
            
            tmp_b.s[j] = bodies[k].s[j]
            mid_point[k].s[j] = 0.5*(tmp_b.s[j] - bodies[k].s[j])

    for k in range(nbodies):
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, mid_point, nbodies, order)  
    
    for k in range(nbodies):
                
        for j in range(3):
            bodies[k].q[j] += dt*g[k][3+j]
            bodies[k].p[j] -= dt*g[k][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution  
    
    _free(mid_point)

    for k in range(nbodies): 
        for j in range(3):    
        
            tmp.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp.p[j] = bodies[k].p[j] - start[k].p[j]
            
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
            '''

            D[k][j+3] = tmp.q[j]*tmp.q[j] + dt2*dt2
            D[k][j] = tmp.p[j]*tmp.p[j] + dt2*dt2
                                    
    _free(start)
    #_free(K)
        
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
    
#'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _one_step_eu(body_t *bodies, unsigned int nbodies, long double dt, int order):
    
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
        
    #cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    #if K == NULL:
        #raise MemoryError 

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))

    _gradients(g, bodies, nbodies, order)


    for k in range(nbodies):
        start[k] = bodies[k]
   

    for k in range(nbodies):
        mass = bodies[k].mass
                
        for j in range(3):
            bodies[k].q[j] += dt*g[k][3+j]
            bodies[k].p[j] -= dt*g[k][j]
#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution


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

            D[k][j+3] = tmp.q[j] + dt
            D[k][j] = tmp.p[j] + dt
                     
    _free(start)
    #_free(K)
        
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
cdef _one_step_rk(body_t *bodies, unsigned int nbodies, long double dt, int order):

    cdef unsigned int i,j,k
    cdef long double dt2 = 0.5*dt

    # variables for "test" implementations
    #'''
    cdef body_t *k1 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k1 == NULL:
        raise MemoryError
    
    cdef body_t *k2 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k2 == NULL:
        raise MemoryError
    
    cdef body_t *k3 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k3 == NULL:
        raise MemoryError
    
    cdef body_t *k4 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k4 == NULL:
        raise MemoryError

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

    _gradients(g, bodies, nbodies, order)

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError

    for k in range(nbodies):
        start[k] = bodies[k]       

    #'''

    #'''
    cdef body_t *k5 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k5 == NULL:
        raise MemoryError
    
    cdef body_t *k6 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k6 == NULL:
        raise MemoryError

    #'''
   


    # Runke-Kutta-Fehlberg (6° order) -- test
    #'''
    for k in range(nbodies):      
            
        mass = bodies[k].mass        
        tmp_q[k].mass = mass
        tmp_p[k].mass = mass     

        #k1   
        for j in range(3):
            
            k1[k].q[j] = dt*g[k][3+j] #dt2*g[k][3+j] 
            k1[k].p[j] = -dt*g[k][j] #-dt2*g[k][j] # 
                     
            tmp_q[k].q[j] = bodies[k].q[j] + 0.25*k1[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]     
                 
            tmp_p[k].q[j] = bodies[k].q[j]  
            tmp_p[k].p[j] = bodies[k].p[j] + 0.25*k1[k].p[j]

    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))   
                    
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)

    for k in range(nbodies): 
        #k2
        for j in range(3):
            
            k2[k].q[j] = dt*0.25*g_q[k][3+j] #dt2*g_q[k][3+j]  
            k2[k].p[j] = -dt*0.25*g_p[k][j]  #-dt2*g_p[k][j]        
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 3./32.*k1[k].q[j] + 9./32.*k2[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]   
                 
            tmp_p[k].q[j] = bodies[k].q[j]  
            tmp_p[k].p[j] = bodies[k].p[j] + 3./32.*k1[k].p[j] + 9./32.*k2[k].p[j]                
 

    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))   
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):                       
        #k3
        for j in range(3):
            
            k3[k].q[j] = 3./8.*dt*g_q[k][3+j] #dt2*g_q[k][3+j]  
            k3[k].p[j] = -3./8.*dt*g_p[k][j] #- dt2*g_p[k][j]    
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 1932./2197.*k1[k].q[j] - 7200./2197.*k2[k].q[j] +  7296./2197.*k3[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]     
                 
            tmp_p[k].q[j] = bodies[k].q[j]   
            tmp_p[k].p[j] = bodies[k].p[j] + 1932./2197.*k1[k].p[j] - 7200./2197.*k2[k].p[j] +  7296./2197.*k3[k].p[j]               
 

    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))   
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k4
        for j in range(3):
            
            k4[k].q[j] = (12./13.*dt)*g_q[k][3+j] #dt2*g_q[k][3+j]     
            k4[k].p[j] = -(12./13.*dt)*g_p[k][j] #-dt2*g_p[k][j]        
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 439./216.*k1[k].q[j] - 8*k2[k].q[j] +  3680./513.*k3[k].q[j] - 845./4104.*k4[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]     
                 
            tmp_p[k].q[j] = bodies[k].q[j] 
            tmp_p[k].p[j] = bodies[k].p[j] + 439./216.*k1[k].p[j] - 8*k2[k].p[j] +  3680./513.*k3[k].p[j] - 845./4104.*k4[k].p[j]                  


    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))   
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k5
        for j in range(3):
            
            k5[k].q[j] = dt*g_q[k][3+j] #dt2*g_q[k][3+j]     
            k5[k].p[j] = -dt*g_p[k][j] #-dt2*g_p[k][j]           
                        
            tmp_q[k].q[j] = bodies[k].q[j] - 8./27.*k1[k].q[j] + 2*k2[k].q[j] -  3544./2565.*k3[k].q[j] + 1859./4104.*k4[k].q[j] - 11./40.*k5[k].q[j]   
            tmp_q[k].p[j] = bodies[k].p[j]     
                 
            tmp_p[k].q[j] = bodies[k].q[j]   
            tmp_p[k].p[j] = bodies[k].p[j] - 8./27.*k1[k].p[j] + 2*k2[k].p[j] -  3544./2565.*k3[k].p[j] + 1859./4104.*k4[k].p[j] - 11./40.*k5[k].p[j]                     


    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))   
          
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)    

    for k in range(nbodies):           
        #k6
        for j in range(3):
            
            k6[k].q[j] = dt2*g_q[k][3+j]   
            k6[k].p[j] = -dt2*g_p[k][j]         
            
            bodies[k].q[j] += (16./135.)*k1[k].q[j] + (6656./12825.)*k3[k].q[j] + (28561./56430.)*k4[k].q[j] - (9./50.)*k5[k].q[j] + (2./55.)*k6[k].q[j]            
            bodies[k].p[j] += (16./135.)*k1[k].p[j] + (6656./12825.)*k3[k].p[j] + (28561./56430.)*k4[k].p[j] - (9./50.)*k5[k].p[j] + (2./55.)*k6[k].p[j]       
   #'''   


    #standard Runke-Kutta (4° order) -- test
    '''              
    for k in range(nbodies):      
            
        mass = bodies[k].mass        
        tmp_q[k].mass = mass
        tmp_p[k].mass = mass               

        #k1   
        for j in range(3):
            
            k1[k].q[j] = dt*g[k][3+j]
            k1[k].p[j] = -dt*g[k][j] 
                     
            tmp_q[k].q[j] = bodies[k].q[j] + 0.5*k1[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j] 
                 
            tmp_p[k].q[j] = bodies[k].q[j]  
            tmp_p[k].p[j] = bodies[k].p[j] + 0.5*k1[k].p[j]

    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))                    
                 
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)

    for k in range(nbodies): 
        #k2
        for j in range(3):
            
            k2[k].q[j] = dt2*g_q[k][3+j]   
            k2[k].p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + 0.5*k2[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]    
                 
            tmp_p[k].q[j] = bodies[k].q[j]
            tmp_p[k].p[j] = bodies[k].p[j] + 0.5*k2[k].p[j]                 


    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))    
          
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):                       
        #k3
        for j in range(3):
            
            k3[k].q[j] = dt2*g_q[k][3+j]   
            k3[k].p[j] = -dt2*g_p[k][j]         
                        
            tmp_q[k].q[j] = bodies[k].q[j] + k3[k].q[j]    
            tmp_q[k].p[j] = bodies[k].p[j]     
                 
            tmp_p[k].q[j] = bodies[k].q[j]  
            tmp_p[k].p[j] = bodies[k].p[j] + k3[k].p[j]                 

    for k in range(nbodies):                                
        memset(g_q[k], 0, 6*sizeof(long double))   
        memset(g_p[k], 0, 6*sizeof(long double))    
           
    _gradients(g_q, tmp_q, nbodies, order)      
    _gradients(g_p, tmp_p, nbodies, order)          

    for k in range(nbodies):           
        #k4
        for j in range(3):
            
            k4[k].q[j] = dt*g_q[k][3+j]   
            k4[k].p[j] = -dt*g_p[k][j]

            bodies[k].q[j] += (1./6.)*k1[k].q[j] + (1./3.)*k2[k].q[j] + (1./3.)*k3[k].q[j] + (1./6.)*k4[k].q[j]   
            bodies[k].p[j] += (1./6.)*k1[k].p[j] + (1./3.)*k2[k].p[j] + (1./3.)*k3[k].p[j] + (1./6.)*k4[k].p[j]
    ''' 


    #standard Runke-Kutta (4° order)
    '''     
    cdef body_t *k1 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k1 == NULL:
        raise MemoryError
    
    cdef body_t *k2 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k2 == NULL:
        raise MemoryError
    
    cdef body_t *k3 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k3 == NULL:
        raise MemoryError
    
    cdef body_t *k4 = <body_t *>malloc(nbodies*sizeof(body_t))
    if k4 == NULL:
        raise MemoryError

    cdef long double **g = <long double **>malloc(nbodies*sizeof(long double *))    
    if g == NULL:
        raise MemoryError    

    for i in range(nbodies):
        g[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if g[i] == NULL:
            raise MemoryError
        memset(g[i], 0, 6*sizeof(long double))

    cdef body_t *tmp = <body_t *>malloc(nbodies*sizeof(body_t))
    if tmp == NULL:
        raise MemoryError

    cdef body_t *start = <body_t *>malloc(nbodies*sizeof(body_t))
    if start == NULL:
        raise MemoryError

    for k in range(nbodies):
        start[k] = bodies[k]       

    #cdef body_t *K = <body_t *>malloc(nbodies*sizeof(body_t))
    #if K == NULL:
        #raise MemoryError 

    for k in range(nbodies):
        start[k] = bodies[k]
 
    _gradients(g, bodies, nbodies, order)
   
    for k in range(nbodies):      
            
        mass = bodies[k].mass        
        tmp.mass = mass
        #tmp_p[k].mass = mass    
        
        #k1   
        for j in range(3):
            
            k1[k].q[j] = dt*g[k][3+j]
            k1[k].p[j] = dt*g[k][j] 
                     
            tmp.q[j] = bodies[k].q[j] + 0.5*k1[k].q[j]
            tmp.p[j] = bodies[k].p[j] - 0.5*k1[k].p[j]  

        # update the gradient

    for k in range(nbodies): 
        memset(g[k], 0, 6*sizeof(long double))     
    _gradients(g, tmp, nbodies, order)                     

    for k in range(nbodies): 
        #k2
        for j in range(3):
            
            k2[k].q[j] = dt2*g[k][3+j]   
            k2[k].p[j] = dt2*g[k][j]         
                        
            tmp.q[j] = bodies[k].q[j] + 0.5*k2[k].q[j]    
            tmp.p[j] = bodies[k].p[j] - 0.5*k2[k].p[j]    

    for k in range(nbodies):                                
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp, nbodies, order)                           

    for k in range(nbodies):                       
        #k3
        for j in range(3):
            
            k3[k].q[j] = dt2*g[k][3+j]   
            k3[k].p[j] = dt2*g[k][j]         
                        
            tmp.q[j] = bodies[k].q[j] + k3[k].q[j]  
            tmp.p[j] = bodies[k].p[j] - k3[k].p[j]                 

    for k in range(nbodies): 
        memset(g[k], 0, 6*sizeof(long double))
    _gradients(g, tmp, nbodies, order)           

    for k in range(nbodies):           
        #k4
        for j in range(3):
            
            k4[k].q[j] = dt*g[k][3+j]   
            k4[k].p[j] = dt*g[k][j]

            bodies[k].q[j] += (1./6.)*k1[k].q[j] + (1./3.)*k2[k].q[j] + (1./3.)*k3[k].q[j] + (1./6.)*k4[k].q[j]
            bodies[k].p[j] += -((1./6.)*k1[k].p[j] + (1./3.)*k2[k].p[j] + (1./3.)*k3[k].p[j] + (1./6.)*k4[k].p[j])

    '''

    for i in range(nbodies):
        free(g[i])
        free(g_p[i])
        free(g_q[i])
        
    free(g);
    free(g_p);
    free(g_q);

    free(k1)
    free(k2)
    free(k3)
    free(k4)
    free(k5)
    free(k6)

    _free(tmp_q)
    _free(tmp_p)
    #_free(tmp)

    cdef body_t tmp_cond

    cdef long double **D = <long double **>malloc(nbodies*sizeof(long double *))    
    if D == NULL:
        raise MemoryError
        
    for i in range(nbodies):
        D[i] = <long double *>malloc(6*sizeof(long double)) #FIXME: for the spins
        if D[i] == NULL:
            raise MemoryError
        memset(D[i], 0, 6*sizeof(long double))


    for k in range(nbodies): 
        for j in range(3):    
        
            tmp_cond.q[j] = bodies[k].q[j] - start[k].q[j]
            tmp_cond.p[j] = bodies[k].p[j] - start[k].p[j]

            '''
            if (tmp.q[j] == 0):
                K[k].q[j] = 0
   
            if (tmp.p[j] == 0):
                K[k].p[j] = 0

            if (tmp.q[j] != 0):
                K[k].q[j] = dt2/(tmp_cond.p[j]*tmp_cond.p[j])
                
            if (tmp.p[j] != 0):
                K[k].p[j] = dt2/(tmp_cond.q[j]*tmp_cond.q[j])

            if (K[k].q[j] > 0.5):
                dt2 = tmp_cond.p[j]*tmp_cond.p[j]*0.5

            if (K[k].p[j] > 0.5):
                dt2 = tmp_cond.q[j]*tmp_cond.q[j]*0.5
            '''

            #D[k][j+3] = tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j] + dt2*dt2*dt2*dt2 #4th order
            D[k][j+3] = tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j]*tmp_cond.q[j] + dt2*dt2*dt2*dt2*dt2*dt2 #6th order

            #D[k][j] = tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j] + dt2*dt2*dt2*dt2 #4th order
            D[k][j] = tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j]*tmp_cond.p[j] + dt2*dt2*dt2*dt2*dt2*dt2 #6th order 

    _free(start)
    #_free(K)
 
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

    #cdef list t_sim = []
    cdef list D_tmp = [[0 for u in range(6)] for k in range(n)]
    cdef list D = []

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
    cdef long double dt2_tmp = 0.
    cdef list t_sim = []

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
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp = _one_step_eu(bodies, n, dt, order)
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_lp(bodies, n, dt, order)
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_sv(bodies, n, dt, order) #FIXME
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_lw(bodies, n, dt, order) #FIXME
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_rk(bodies, n, dt, order)
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_gar(bodies, n, dt, order) #FIXME
        dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_icn(bodies, n, dt, order, ICN_it)
        #dx, dy, dz, dpx, dpy, dpz, dt2_tmp  = _one_step_icn_mod(bodies, n, dt, order, ICN_it)

        time += dt2_tmp    
        # store 1 every nthin steps 

        if ( (i+1)%nthin == 0.):    
            solution.append([bodies[j] for j in range(n)])
            h, t, v = _hamiltonian(bodies, n, order)

            for k in range(n):
                D_tmp[k][0] = dx[k] 
                D_tmp[k][1] = dy[k]  
                D_tmp[k][2] = dz[k] 
                D_tmp[k][3] = dpx[k]  
                D_tmp[k][4] = dpy[k]
                D_tmp[k][5] = dpz[k] 
        
            H.append(h)
            T.append(t)
            V.append(v)
            D.append(D_tmp) 
            t_sim.append(time) 
         
        # divide in files with buffer_lenght steps each    
        if ( (i+1)%buffer_length == 0.):
        
            pickle.dump(solution, open('solution_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(T, open('kinetic_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(V, open('potential_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(H, open('hamiltonian_{}_order{}.pkl'.format(n_sol, order),'wb'))

            pickle.dump(t_sim, open('time_{}_order{}.pkl'.format(n_sol, order),'wb'))
            pickle.dump(D, open('error_{}_order{}.pkl'.format(n_sol, order),'wb'))

            n_sol += 1
            H        = []
            T        = []
            V        = []
            solution = []
            D = []
            t_sim = []

        '''
        if ( (i+1)%(nthin*plot_step) == 0):

            f_index = (i+1)/(nthin*plot_step) - 1         
            t_sim.append(time)     
                
            pickle.dump(t_sim, open('error_{}_order{}.pkl'.format(f_index, order),'wb'))

            for k in range(n):
                D[f_index][k][0] = dx[k] 
                D[f_index][k][1] = dy[k]  
                D[f_index][k][2] = dz[k] 
                D[f_index][k][3] = dpx[k]  
                D[f_index][k][4] = dpy[k]
                D[f_index][k][5] = dpz[k] 

            pickle.dump(t_sim, open('error_{}_order{}.pkl'.format(f_index, order),'wb'))
        ''' 

    return #() D, t_sim)
