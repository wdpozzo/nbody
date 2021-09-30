import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, abs
from nbody.body cimport body_t, system_t

cdef double G = 1.0#6.67e-11
cdef double C = 3.0e8
cdef double Msun = 2e30
cdef double GM = 1.32712440018e20

cdef inline double _modulus(double x, double y, double z) nogil:
    return x*x+y*y+z*z

cdef double _dot(double *v1, double *v2) nogil:
    cdef unsigned int i
    cdef double result = 0.0
    
    for i in range(3):
        result += v1[i]*v2[i]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _hamiltonian(body_t *bodies, unsigned int N, int order):
    cdef unsigned int i, j, k
    cdef double T = 0.0
    cdef double V = 0.0
    cdef double H = 0.0
    
    # for 1 PN
    cdef double[3] normal
    cdef double p2
    cdef double p4
    cdef double r
    cdef double mi, mj
    cdef double V0
    cdef double C2 = C*C
    
    if order == 0:
        # compute the kinetic part
        for i in range(N):
            T += _kinetic_energy(bodies[i])
            # and the potential
            for j in range(i+1,N):
                V += _potential_0pn(bodies[i], bodies[j])
        return T+V
    
    cdef double n_pi

    if order == 1:
        # compute the kinetic part
        for i in range(N):
            mi = bodies[i].mass
            T += _kinetic_energy(bodies[i])
            p2 = _modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])
            p4 = p2*p2
            T += (-(1./8.)*p4/mi**3)/C**2

            # precompute a dot product
            n_pi = _dot(normal,bodies[i].p)
            
            # and the potential
            for j in range(i+1,N):
            
                for k in range(3):
                    normal[k] = (bodies[i].q[k]-bodies[j].q[k])/r
            
                mj = bodies[j].mass
                r  = sqrt(_modulus(bodies[i].q[0]-bodies[j].q[0],
                                   bodies[i].q[1]-bodies[j].q[1],
                                   bodies[i].q[2]-bodies[j].q[2]))
                
                V0 = _potential_0pn(bodies[i], bodies[j])
                V += V0
                    
                V += ((1./8.)*(2.0*V0))*(-12.*p2/(mi*mi)+14.0*_dot(bodies[i].p,bodies[j].p)/(mi*mj)+2.0*n_pi*_dot(normal,bodies[j].p))/C2
                #print('2',i,j,V)
                V += (0.25*V0*G*(mi+mj)/r)/C2
                #print('3',i,j,V)
        return T+V

cdef inline double _kinetic_energy(body_t b) nogil:
    return 0.5*_modulus(b.p[0],b.p[1],b.p[2])/b.mass

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _potential(body_t *bodies, unsigned int N, int order) nogil:
    
    cdef unsigned int i,j
    cdef double V = 0.0

    for i in range(N):
        for j in range(i+1,N):
            V = _potential_0pn(bodies[i], bodies[j])
#    if order >= 1:
#        V = _potential_1pn(bi, bj)
#
#    if order >= 2:
#        V += _potential_2pn(bi, bj)
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
cdef double _potential_0pn(body_t b1, body_t b2) nogil:
                            
    cdef double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    
    return -0.5*G*b1.mass*b2.mass/r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[double, mode="c", ndim=2] _gradients(body_t *bodies, unsigned int N, int order):

    cdef unsigned int i,j,k
    cdef np.ndarray[double, mode="c", ndim=2] force = np.zeros((N,6),dtype=np.double)
    cdef double[:,:] force_view = force
    cdef np.ndarray[double, mode="c", ndim=1] tmp = np.zeros(6,dtype=np.double)
    cdef double[:] tmp_view = tmp
    
    for i in range(N):

        for j in range(N):
            if i != j:
                tmp_view = _gradient(bodies[i],bodies[j],order)
                for k in range(6):
                    force_view[i,k] = force_view[i,k]+tmp_view[k]

    return force

cdef double[:] _gradient(body_t b1, body_t b2, int order):
    cdef unsigned int k
    cdef double[:] f

    if order == 0:
        f = _gradient_0pn(b1,b2)

    elif order == 1:
        f = _gradient_1pn(b1,b2)
#
#    if order >= 2:
#        f += _gradient_2pn(b1,b2)
#
#    if order >= 3:
#        f += _gradient_3pn(b1,b2)
#
#    if order >= 4:
#        raise(NotImplementedError)
    
    return f

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:] _gradient_0pn(body_t b1, body_t b2):
    
    cdef double dx = b1.q[0]-b2.q[0]
    cdef double dy = b1.q[1]-b2.q[1]
    cdef double dz = b1.q[2]-b2.q[2]
    cdef double r  = sqrt(_modulus(dx,dy,dz))

    cdef double prefactor = -0.5*G*b1.mass*b2.mass/(r*r*r)
    cdef double[6] f

    f[0] = prefactor*dx
    f[1] = prefactor*dy
    f[2] = prefactor*dz
    f[3] = b1.p[0]/b1.mass
    f[4] = b1.p[1]/b1.mass
    f[5] = b1.p[2]/b1.mass
    
    return f


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:] _gradient_1pn(body_t b1, body_t b2):
    
    cdef unsigned int k
    cdef double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef double r2 = r*r
    cdef double r3 = r2*r
    cdef double[3] normal

    for k in range(3):
        normal[k] = (b1.q[k]-b2.q[k])/r

    cdef double m1 = b1.mass
    cdef double m2 = b2.mass
    
    cdef double m1sq = m1*m1
    cdef double m1cu = m1*m1sq
    
    cdef double m1m2 = m1*m2
    
    cdef double p2 = _modulus(b1.p[0],b1.p[1],b1.p[2])
    cdef double p4 = p2*p2

    cdef double V0 = -G*b1.mass*b2.mass/r
    cdef double dV0
    cdef double C2 = C*C
    
    cdef double n_p1 = _dot(normal,b1.p)
    cdef double n_p2 = _dot(normal,b2.p)
    cdef double p1_p2 = _dot(b1.p,b2.p)
    
    cdef double prefactor = -G*m1*m2/r3
    cdef double[6] f

    for k in range(3):

        dV0  = prefactor*(b1.q[k]-b2.q[k])
        # derivative wrt p
        f[k] = 0.5*dV0 + (0.125*dV0*(-12.*p2/(m1sq)+14.0*p1_p2/(m1m2)+2.0*n_p1*n_p2)+(1./8.)*V0*(2.*((r2-(b1.q[k]-b2.q[k])**2+b1.q[k]*b2.q[k])/(m1m2*r3))*(b1.p[k]*n_p2+b2.p[k]*n_p1)+0.25*G*(m1+m2)*(dV0/r-V0/r3) )) / C2

        # derivative wrt q
        f[3+k] = b1.p[k]/m1+ (-(1./(8.*m1cu))*4.0*b1.p[k]*p2+(1./8.)*V0*(-24.0*b1.p[k]*p2/m1sq+14.0*b2.p[k]/(m1m2)+2.0*normal[k]*n_p2/(m1m2)))/C2
    
    return f
