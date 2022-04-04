import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, abs
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from nbody.body cimport body_t, merger, _merge_bodies


import astropy.units as u

cdef long double G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) 

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

cdef long double C = 299792458. #*(u.meter/u.second)
cdef long double Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

cdef long double _modulus(long double x, long double y, long double z) nogil:

    return x*x+y*y+z*z
                              
cdef long double _dot(long double *v1, long double *v2) nogil:
    cdef unsigned int i
    cdef long double result = 0.0
    
    for i in range(3):
        result += v1[i]*v2[i]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef (long double, long double, long double) _hamiltonian(body_t *bodies, unsigned int N, int order) nogil:

    cdef unsigned int i, j, k
    cdef long double T = 0.0
    cdef long double V = 0.0
    cdef long double H = 0.0
    
    cdef long double mi
    
    
    cdef long double C2 = C*C
    cdef long double C4 = C2*C2
    
    V = _potential(bodies, N, order)
    
    if order == 0:
        # compute the kinetic part
        for i in range(N):
        
            T += _kinetic_energy(bodies[i])

    if order >= 1:    
        # compute the kinetic part
        for i in range(N):
        
            mi = bodies[i].mass
            
            #T += _kinetic_energy(bodies[i])
            T -= ((1./8.)*(_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])*_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2]))/(mi*mi*mi))/C2

    if order >= 2:

        # compute the kinetic part
        for i in range(N):
            mi = bodies[i].mass
           
            #T += _kinetic_energy(bodies[i])   
            #T -= (1./8.)*(_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])*_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2]))/(mi*mi*mi)/(C2) 
            T += ((1./16.)*(_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])*_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])*_modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2]))/(mi*mi*mi*mi*mi))/C4
         
    H = T + V
        
    return (H, T, V)

 
cdef inline long double _kinetic_energy(body_t b) nogil:

    cdef long double m = b.mass
    cdef long double num = 0.5*_modulus(b.p[0],b.p[1],b.p[2])
    
    #print(num, m)
    
    return (num*1.0)/(m*1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long double _potential(body_t *bodies, unsigned int N, int order) nogil:
    
    cdef unsigned int i,j
    cdef long double V = 0.0

    for i in range(N):
    
        for j in range(i+1,N):
        #for j in range(N):
        
            #if i != j :
                #continue   
                
            V += _potential_0pn(bodies[i], bodies[j])
            
            if order >= 1:
                V += _potential_1pn(bodies[i], bodies[j])

            if order >= 2:
                V += _potential_2pn(bodies[i], bodies[j])
#
#    if order >= 3:
#        V += _potential_3pn(bi, bj)
#
#    if order >= 4:
#        raise(NotImplementedError)

    return V

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long double _potential_0pn(body_t b1, body_t b2) nogil:
                            
    cdef long double r  = sqrt(_modulus(b1.q[0]-b2.q[0], b1.q[1]-b2.q[1], b1.q[2]-b2.q[2]))

    return - G*b1.mass*b2.mass/r
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long double _potential_1pn(body_t b1, body_t b2) nogil:
                            
    cdef long double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef long double V = 0.0
    cdef long double m1 = b1.mass
    cdef long double m2 = b2.mass
    cdef long double m1m2 = m1*m2
    cdef long double *normal = <long double *>malloc(3*sizeof(long double))
    cdef long double p12 = _modulus(b1.p[0],b1.p[1],b1.p[2])

    cdef long double C2 = C*C
    
    for k in range(3):
        normal[k] = (b1.q[k]-b2.q[k])/r
        
    V += ((1./8.)*(G*m1m2/r)*(-12.*p12/(m1*m1) + 14.0*_dot(b1.p, b2.p)/m1m2 + 2.0*_dot(normal,b1.p)*_dot(normal,b2.p)/m1m2))/C2

    V += (0.25*(G*m1m2/r)*G*(m1+m2)/r)/C2
    
    free(normal)
    
    return V

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long double _potential_2pn(body_t b1, body_t b2) nogil:
                    
    cdef long double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef long double r2 = r*r
    cdef long double m1 = b1.mass
    cdef long double m2 = b2.mass
    cdef long double m1m2 = m1*m2
    cdef long double m1m2sq = m1m2*m1m2    
    cdef long double m1sq = m1*m1
    cdef long double m2sq = m2*m2
    cdef long double m1cu = m1*m1sq
    cdef long double m1qu = m1*m1cu           

    cdef long double p1_p2 = _dot(b1.p,b2.p)            
    cdef long double p12 = _modulus(b1.p[0],b1.p[1],b1.p[2])
    cdef long double p22 = _modulus(b2.p[0],b2.p[1],b2.p[2]) 
    cdef long double p14 = p12*p12
               
    cdef long double V = 0.0 
    cdef long double *normal = <long double *>malloc(3*sizeof(long double))
    
    cdef long double C2 = C*C
    cdef long double C4 = C2*C2
    
    for k in range(3):
        normal[k] = (b1.q[k]-b2.q[k])/r
          
    V += ((1./8.)*(G*m1m2/r)*(5.*p14/m1qu - (11./2.)*p12*p22/m1m2sq - (p1_p2)*(p1_p2)/m1m2sq + 5.*(p12*_dot(normal,b2.p)*_dot(normal,b2.p))/m1m2sq - 6.*(p1_p2*_dot(normal,b1.p)*_dot(normal,b2.p))/m1m2sq - (3./2.)*(_dot(normal,b1.p)*_dot(normal,b1.p))*(_dot(normal,b2.p)*_dot(normal,b2.p))/m1m2sq) + (1./4.)*(G*G*m1m2/r2)*(m2*(10.*p12/m1sq + 19.*p22/m2sq) - (1./2.)*(m1+m2)*(27.*p1_p2 + 6.*_dot(normal,b1.p)*_dot(normal,b2.p))/m1m2 ))/C4
    
    V -= ((1./8.)*(G*m1m2/r)*G*G*(m1sq+5.*m1m2+m2sq)/r2)/C4
    
    free(normal)
    
    return V

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradients(long double **out, body_t *bodies, unsigned int N, int order) nogil:

    cdef unsigned int i,j,k
    cdef long double *tmp = <long double *>malloc(6*sizeof(long double))
    #cdef long double *tmp = np.zeros(6, dtype = np.longdouble)    
    #memset(tmp, 0, 6*sizeof(long double))
     
    
    for i in range(N):
    
        _gradient_free_particle(tmp, bodies[i])
        
        for k in range(6):
            out[i][k] += tmp[k] 
               
        #for j in range(N):
            #if i != j:
                #continue
        '''
        #test
          
        if i==0:
            j = 1            
            _gradient(tmp, bodies[i], bodies[j], order)
                       
        if i==1:
            j = 0
            _gradient(tmp, bodies[i], bodies[j], order)
        
        for k in range(6):    
            out[i][k] += tmp[k]
        '''
        
        for j in range(N):
        
            if i == j:
                continue
      
            _gradient(tmp, bodies[i], bodies[j], order)
            
            for k in range(6):            
                out[i][k] += tmp[k]
        
    free(tmp)
      
    return

cdef void _gradient(long double *out, body_t b1, body_t b2, int order) nogil:
    
    _gradient_0pn(out, b1, b2)

    if order >= 1:
        _gradient_1pn(out, b1, b2)

    if order >= 2:
        _gradient_2pn(out, b1, b2)

#    if order >= 3:
#        f += _gradient_3pn(b1,b2)
#
#    if order >= 4:
#        raise(NotImplementedError)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradient_free_particle(long double *out, body_t b1) nogil:

    # first 3 elements are the derivative wrt to q
    
    out[0] = 0.
    out[1] = 0.
    out[2] = 0.   
    
    # second 3 elements are the derivative wrt p

    out[3] = b1.p[0]/b1.mass
    out[4] = b1.p[1]/b1.mass
    out[5] = b1.p[2]/b1.mass  
    
    '''
    out[3] = 0.
    out[4] = 0.
    out[5] = 0.
    '''
    
    return
    
"""
We are going to include the minus sign in Hamilton equations directly in the gradient
"""
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradient_0pn(long double *out, body_t b1, body_t b2) nogil:
    
    cdef long double dx = b1.q[0]-b2.q[0]
    cdef long double dy = b1.q[1]-b2.q[1]
    cdef long double dz = b1.q[2]-b2.q[2]
    cdef long double r  = sqrt(_modulus(dx,dy,dz))
    cdef long double r3 = r*r*r
    
    cdef long double prefactor = G*b1.mass*b2.mass/r3
    
    # first 3 elements are the derivative wrt to q
    out[0] = prefactor*dx
    out[1] = prefactor*dy
    out[2] = prefactor*dz

    # second 3 elements are the derivative wrt p
    '''
    out[3] += b1.p[0]/b1.mass
    out[4] += b1.p[1]/b1.mass  
    out[5] += b1.p[2]/b1.mass
    '''

    out[3] = 0.
    out[4] = 0.
    out[5] = 0.

    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradient_1pn(long double *out, body_t b1, body_t b2) nogil:

    cdef unsigned int k
    cdef long double r  = sqrt(_modulus(
    
    b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef long double r2 = r*r
    cdef long double r3 = r2*r
    cdef long double r4 = r3*r
    cdef long double *dq = <long double *>malloc(3*sizeof(long double))
    cdef long double *normal = <long double *>malloc(3*sizeof(long double))

    for k in range(3):
        dq[k]     = b1.q[k]-b2.q[k]
        
        #normal without normalization...why? cause in the following expression the every member is explicit and so the factor 1/r is already counted there
        normal[k] = dq[k] #/r

    cdef long double m1 = b1.mass
    cdef long double m2 = b2.mass
    
    cdef long double m1sq = m1*m1
    cdef long double m1cu = m1*m1sq
    
    cdef long double m1m2 = m1*m2
    
    cdef long double p12 = _modulus(b1.p[0],b1.p[1],b1.p[2])
    cdef long double p22 = _modulus(b2.p[0],b2.p[1],b2.p[2])

    #cdef long double prefactor = 0.5*G*m1*m2/r3

    cdef long double C2 = C*C
    
    cdef long double n_p1 = _dot(normal,b1.p)
    cdef long double n_p2 = _dot(normal,b2.p)
    cdef long double dq_p1 = _dot(dq,b1.p)
    cdef long double dq_p2 = _dot(dq,b2.p)
    cdef long double p1_p2 = _dot(b1.p,b2.p)
    
    for k in range(3):

        # derivative wrt q
        #out[k] += ( - 0.5*G*G*m1m2*(m1 + m2)*dq[k]/r4 - 0.125*G*m1m2*dq[k]*( 14*p1_p2/m1m2 + 2*(n_p1)*(n_p2)/(m1m2*r2) + (-12*p12)/m1sq)/r3 + 0.25*G*m1m2*(b1.p[k]*(n_p2)/(m1m2*r2) + 2*b2.p[k]*(n_p1)/(m1m2*r2) - 4*dq[k]*(n_p1)*(n_p2)/(m1m2*r4))/r)/C2 # 1PN order
       
        out[k] += ( -0.5*G*G*m1m2*(m1 + m2)*dq[k]/r4 - 0.125*G*m1m2*dq[k]*(14*p1_p2/m1m2 + 2*n_p1*n_p2/(m1m2*r2) -12.*p12/m1sq)/r3 + 0.125*G*m1m2*(2*b1.p[k]*n_p2/(m1m2*r2) + 2*b2.p[k]*n_p1/(m1m2*r2) - 4.*dq[k]*n_p1*n_p2/(m1m2*r4))/r )/C2

        # derivative wrt p
        #out[3+k] +=  (0.125*G*m1m2*(14*b2.p[k]/m1m2 + 2*dq[k]*(n_p2)/(m1m2*r2) - 24*b1.p[k]/m1sq)/r - 0.5*b1.p[k]*(p12)/m1cu )/C2 #1PN order
        
        out[k+3] += ( 0.125*G*m1m2*(14*b2.p[k]/m1m2 + 2*dq[k]*n_p2/(m1m2*r2) - 24*b1.p[k]/m1sq)/r - 0.5*b1.p[k]*p12/m1cu )/C2
    
    free(dq)
    free(normal)
    
    return
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradient_2pn(long double *out, body_t b1, body_t b2) nogil:
    
    cdef unsigned int k
    cdef long double r = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef long double r2 = r*r
    cdef long double r3 = r2*r
    cdef long double r4 = r3*r
    cdef long double r5 = r4*r
    cdef long double r6 = r5*r
    cdef long double *dq = <long double *>malloc(3*sizeof(long double))
    cdef long double *normal = <long double *>malloc(3*sizeof(long double))

    for k in range(3):
        dq[k] = b1.q[k]-b2.q[k]
        
        #normal without normalization...why? cause in the following expression the "1/r" factor in every member is explicit and so the factor 1/r is already counted there
        normal[k] = dq[k] #/r

    cdef long double m1 = b1.mass
    cdef long double m2 = b2.mass
    
    cdef long double m1sq = m1*m1
    cdef long double m2sq = m2*m2
    cdef long double m1cu = m1*m1sq
    cdef long double m1qu = m1*m1cu
    cdef long double m1fi = m1*m1qu
    
    cdef double m1m2 = m1*m2
    cdef double m1m2sq = m1m2*m1m2
    
    cdef long double p12 = _modulus(b1.p[0],b1.p[1],b1.p[2])
    cdef long double p22 = _modulus(b2.p[0],b2.p[1],b2.p[2])

    #cdef long double prefactor = 0.5*G*m1*m2/r3

    cdef long double C2 = C*C
    cdef long double C4 = C2*C2
    
    cdef long double n_p1 = _dot(normal,b1.p)
    cdef long double n_p2 = _dot(normal,b2.p)
    cdef long double p1_p2 = _dot(b1.p,b2.p)

    for k in range(3):
        
        # derivative wrt q
        
        #out[k] += ( 0.125*G*G*G*m1m2*3*dq[k]*(m1sq + 5*m1m2 + m2sq)/r5 - 0.5*G*G*m1m2*dq[k]*(m2*(19*p22/m2sq + 10*p12/m1sq) - (0.5*m1 + 0.5*m2)*(27*p1_p2 + 6*n_p1*n_p2/r2)/m1m2)/r4 - 0.25*G*G*(0.5*m1 + 0.5*m2)*(6*b1.p[k]*n_p2/r2 + 6*b2.p[k]*n_p1/r2 - 12*dq[k]*n_p1*n_p2/r4)/r2 - 0.125*G*m1m2*dq[k]*(5*p12*p12*n_p2*n_p2/(m1m2sq*r2) - 5.5*p12*p22/m1m2sq - p1_p2*p1_p2/m1m2sq - 6*p1_p2*n_p1*n_p2/m1m2sq - 1.5*n_p1*n_p1*n_p2*n_p2/(m1m2sq*r2) + 5*p12*p12/m1qu)/r3 + 0.125*G*m1m2*(-6*b1.p[k]*p1_p2*n_p2/(m1m2sq*r2) - 3.0*b1.p[k]*n_p1*n_p2*n_p2/(m1m2sq*r2) + 10*b2.p[k]*p12*p12*n_p2/(m1m2sq*r2) - 6*b2.p[k]*p1_p2*n_p1/(m1m2sq*r2) - 3.0*b2.p[k]*n_p1*n_p1*n_p2/(m1m2sq*r4) + 6.*dq[k]*n_p1*n_p1*n_p2*n_p2/(m1m2sq*r6) - 10.*dq[k]*p12*p12*n_p2*n_p2/(m1m2sq*r4) + 12.*dq[k]*p1_p2*n_p1*n_p2/(m1m2sq*r4))/r )/C4
        
        out[k] += ( 0.125*G*G*G*m1m2*3*dq[k]*(m1sq + 5*m1m2 + m2sq)/r5 - 0.5*G*G*m1m2*dq[k]*(m2*(19.*p22/m2sq + 10*p12/m1sq) - 0.5*(m1+m2)*(27*p1_p2 + 6*n_p1*n_p2/r2)/m1m2)/r4 - 0.125*G*G*(m1 + m2)*(6*b1.p[k]*n_p2/r2 + 6*b2.p[k]*n_p1/r2 -12.*dq[k]*n_p1*n_p2/r4 )/r2 - 0.125*G*m1m2*dq[k]*(5*p12*n_p2*n_p2/(m1m2sq*r2) - 5.5*p12*p22/m1m2sq - p1_p2*p1_p2/m1m2sq - 6*p1_p2*n_p1*n_p2/(m1m2sq*r2) - 1.5*n_p1*n_p1*n_p2*n_p2/(m1m2sq*r4) + 5*p12*p12/m1qu)/r3 + 0.125*G*m1m2*(-6*b1.p[k]*p1_p2*n_p2/(m1m2sq*r2) - 3.0*b1.p[k]*n_p1*n_p2*n_p2/(m1m2sq*r4) + 10*b2.p[k]*p12*p12*n_p2/(m1m2sq*r2) - 6*b2.p[k]*p1_p2*n_p1/(m1m2sq*r2) - 3.0*b2.p[k]*n_p1*n_p1*n_p2/(m1m2sq*r4) + 6.*dq[k]*n_p1*n_p1*n_p2*n_p2/(m1m2sq*r6) - 10.*dq[k]*p12*p12*p22*p22/(m1m2sq*r4) + 12.*dq[k]*p1_p2*n_p1*n_p2/(m1m2sq*r4))/r )/C4

        # derivative wrt p
        
        #out[k+3] += ( 0.25*G*G*m1m2*(-(0.5*m1 + 0.5*m2)*(27*b2.p[k] + 6*dq[k]*(n_p2)/r2)/m1m2 + 20*m2*b1.p[k]/m1sq)/r2 + 0.125*G*m1m2*(20.0*b1.p[k]*p12*n_p2*n_p2/(m1m2sq*r2) - 11.*b1.p[k]*p12/m1m2sq - 2*b2.p[k]*p1_p2/m1m2sq -6*b2.p[k]*n_p1*n_p2/(m1m2sq*r2) - 6*dq[k]*p1_p2*n_p2/(m1m2sq*r2) - 3.*dq[k]*n_p1*n_p2*n_p2/(m1m2sq*r2) + 20*b1.p[k]*p12/m1qu)/r + 0.375*b1.p[k]*p12*p12/m1fi )/C4
        
        out[k+3] += ( 0.25*G*G*m1m2*(-(0.5*m1 + 0.5*m2)*(27*b2.p[k] + 6*dq[k]*n_p2/r2)/m1m2 + 20*m2*b1.p[k]/m1sq)/r2 + 0.125*G*m1m2*(10*b1.p[k]*p12*n_p2*n_p2/(m1m2sq*r2) - 11.0*b1.p[k]*p22/m1m2sq - 2*b2.p[k]*p1_p2/m1m2sq - 6*b2.p[k]*n_p1*n_p2/(m1m2sq*r2) - 6*dq[k]*p1_p2*n_p2/(m1m2sq*r2) - 3.*dq[k]*n_p1*n_p2*n_p2/(m1m2sq*r4) + 20*b1.p[k]*p12/m1qu)/r + 0.375*b1.p[k]*p12*p12/m1fi )/C4

    free(dq)
    free(normal)
    
    return 
