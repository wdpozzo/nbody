import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, abs
from libc.stdlib cimport malloc, free
from nbody.body cimport body_t

cdef long double G = 1.0#6.67e-11
cdef long double C = 1.0#3.0e8
cdef long double Msun = 2e30
cdef long double GM = 1.32712440018e20

cdef inline long double _modulus(long double x, long double y, long double z) nogil:
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
cdef long double _hamiltonian(body_t *bodies, unsigned int N, int order):
    cdef unsigned int i, j, k
    cdef long double T = 0.0
    cdef long double V = 0.0
    cdef long double H = 0.0
    
    # for 1 PN
    cdef long double[3] normal
    cdef long double p2
    cdef long double p4
    cdef long double r
    cdef long double mi, mj
    cdef long double V0
    cdef long double C2 = C*C
    
    if order == 0:
        # compute the kinetic part
        for i in range(N):
            T += _kinetic_energy(bodies[i])
            # and the potential
            for j in range(i+1,N):
                V += _potential_0pn(bodies[i], bodies[j])
        return T+V
    
    cdef long double n_pi

    if order == 1:
        # compute the kinetic part
        for i in range(N):
            mi = bodies[i].mass
            T += _kinetic_energy(bodies[i])
            
#            print('1 T',i,T)
            p2 = _modulus(bodies[i].p[0],bodies[i].p[1],bodies[i].p[2])
            p4 = p2*p2
            T += (-(1./8.)*p4/mi**3)/C**2
#            print('2 T',i,T)
            # precompute a dot product
            n_pi = _dot(normal,bodies[i].p)
            
            # and the potential
            for j in range(i+1,N):
            
                r  = sqrt(_modulus(bodies[i].q[0]-bodies[j].q[0],
                                   bodies[i].q[1]-bodies[j].q[1],
                                   bodies[i].q[2]-bodies[j].q[2]))
                                   
                for k in range(3):
                    normal[k] = (bodies[i].q[k]-bodies[j].q[k])/r
            
                mj = bodies[j].mass
                
                V0 = _potential_0pn(bodies[i], bodies[j])
                V += V0
#                print('1 V',i,j,V)
                V += ((1./8.)*(2.0*V0))*(-12.*p2/(mi*mi)+14.0*_dot(bodies[i].p,bodies[j].p)/(mi*mj)+2.0*n_pi*_dot(normal,bodies[j].p))/C2
#                print('1 V det',i,j,_dot(bodies[i].p,bodies[j].p),mi*mi,mi*mj,n_pi,_dot(normal,bodies[j].p))
#                print('2 V',i,j,V)
                V += (0.25*V0*G*(mi+mj)/r)/C2
#                print('3 V',i,j,V)
        return T+V

cdef inline long double _kinetic_energy(body_t b) nogil:
    return 0.5*_modulus(b.p[0],b.p[1],b.p[2])/b.mass

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long double _potential(body_t *bodies, unsigned int N, int order) nogil:
    
    cdef unsigned int i,j
    cdef long double V = 0.0

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
cdef long double _potential_0pn(body_t b1, body_t b2) nogil:
                            
    cdef long double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    
    return -0.5*G*b1.mass*b2.mass/r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradients(long double **out, body_t *bodies, unsigned int N, int order) nogil:

    cdef unsigned int i,j,k
    cdef long double *tmp = <long double *>malloc(6*sizeof(long double))
    
    for i in range(N):

        for j in range(N):
            if i != j:
                _gradient(tmp, bodies[i],bodies[j],order)
                for k in range(6):
                    out[i][k] += tmp[k]
    free(tmp)
    return

cdef void _gradient(long double *out, body_t b1, body_t b2, int order) nogil:
    
    _gradient_0pn(out, b1, b2)

    if order >= 1:
        _gradient_1pn(out, b1, b2)
#
#    if order >= 2:
#        f += _gradient_2pn(b1,b2)
#
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
cdef void _gradient_0pn(long double *out, body_t b1, body_t b2) nogil:
    
    cdef long double dx = b1.q[0]-b2.q[0]
    cdef long double dy = b1.q[1]-b2.q[1]
    cdef long double dz = b1.q[2]-b2.q[2]
    cdef long double r  = sqrt(_modulus(dx,dy,dz))

    cdef long double prefactor = 0.5*G*b1.mass*b2.mass/(r*r*r)
    
    # first 3 elements are the derivative wrt to q
    out[0] = prefactor*dx
    out[1] = prefactor*dy
    out[2] = prefactor*dz
    # second 3 elements are the derivative wrt p
    out[3] = b1.p[0]/b1.mass
    out[4] = b1.p[1]/b1.mass
    out[5] = b1.p[2]/b1.mass
    
    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gradient_1pn(long double *out, body_t b1, body_t b2) nogil:
    """
    We are going to include the minus sign in Hamilton equations directly in the gradient
    
    d1PN/dx := 0.25*G**2*m1*m2*(m1 + m2)*(-2*x1 + 2*x2)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 + 0.125*G*m1*m2*(-x1 + x2)*((14*px1*px2 + 14*py1*py2 + 14*pz1*pz2)/(m1*m2) + 2*(px1*(x1 - x2) + py1*(y1 - y2) + pz1*(z1 - z2))*(px2*(x1 - x2) + py2*(y1 - y2) + pz2*(z1 - z2))/(m1*m2*((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) + (-12*px1**2 - 12*py1**2 - 12*pz1**2)/m1**2)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(3/2) + 0.125*G*m1*m2*(2*px1*(px2*(x1 - x2) + py2*(y1 - y2) + pz2*(z1 - z2))/(m1*m2*((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) + 2*px2*(px1*(x1 - x2) + py1*(y1 - y2) + pz1*(z1 - z2))/(m1*m2*((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) + 2*(-2*x1 + 2*x2)*(px1*(x1 - x2) + py1*(y1 - y2) + pz1*(z1 - z2))*(px2*(x1 - x2) + py2*(y1 - y2) + pz2*(z1 - z2))/(m1*m2*((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2))/sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    d1PN/dpx := 0.125*G*m1*m2*(14*px2/(m1*m2) + 2*(x1 - x2)*(px2*(x1 - x2) + py2*(y1 - y2) + pz2*(z1 - z2))/(m1*m2*((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) - 24*px1/m1**2)/sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 0.5*px1*(px1**2 + py1**2 + pz1**2)/m1**3
    """
    cdef unsigned int k
    cdef long double r  = sqrt(_modulus(b1.q[0]-b2.q[0],b1.q[1]-b2.q[1],b1.q[2]-b2.q[2]))
    cdef long double r2 = r*r
    cdef long double r3 = r2*r
    cdef long double[3] dq
    cdef long double[3] normal

    for k in range(3):
        dq[k]     = b1.q[k]-b2.q[k]
        normal[k] = dq[k]/r

    cdef long double m1 = b1.mass
    cdef long double m2 = b2.mass
    
    cdef long double m1sq = m1*m1
    cdef long double m1cu = m1*m1sq
    
    cdef long double m1m2 = m1*m2
    
    cdef long double p1sq = _modulus(b1.p[0],b1.p[1],b1.p[2])

    cdef long double Gmm_r = G*m1*m2/r
    cdef long double C2 = C*C
    
    cdef long double n_p1 = _dot(normal,b1.p)
    cdef long double n_p2 = _dot(normal,b2.p)
    cdef long double dq_p1 = _dot(dq,b1.p)
    cdef long double dq_p2 = _dot(dq,b2.p)
    cdef long double p1_p2 = _dot(b1.p,b2.p)
    
    cdef long double prefactor = -Gmm_r/r2
    cdef long double parenthesis = -12.0*p1sq/m1sq+14.0*p1_p2/m1m2+2.0*n_p1*n_p2/m1m2
    cdef long double[6] f

    for k in range(3):

        # derivative wrt q
        out[k] += (-0.125*prefactor*dq[k]*parenthesis + \
                  0.250*Gmm_r*(n_p2*(b1.p[k]-dq[k]*dq_p1/r2)+n_p1*(b2.p[k]-dq[k]*dq_p2/r2))/(m1m2*r) + \
                  0.25*prefactor*dq[k]*G*(m1+m2)/r + \
                  0.25*Gmm_r*(-G*(m1+m2)*dq[k]/r3))/C2

        # derivative wrt p
        out[3+k] += (-0.5*b1.p[k]*p1sq/m1cu + Gmm_r * \
                    (-24.*b1.p[k]/m1sq + 14.0*b2.p[k]/m1m2 + (2.*dq[k]/(m1m2*r2))*(b2.p[k]*dq_p2)))/C2
        
    return

cdef int merger(body_t b1, body_t b2, double r) nogil:
    if (2*G*b1.mass/(C*C)+2*G*b2.mass/(C*C))< r:
        return 1
    else:
        return 0
    
