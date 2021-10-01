cimport cython
from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _create_body(body_t *b,
                       long double mass,
                       long double x,
                       long double y,
                       long double z,
                       long double px,
                       long double py,
                       long double pz,
                       long double sx,
                       long double sy,
                       long double sz) nogil:
    
    b.mass = mass
    b.q[0]    = x
    b.q[1]    = y
    b.q[2]    = z
    b.p[0]    = px
    b.p[1]    = py
    b.p[2]    = pz
    b.s[0]    = sx
    b.s[1]    = sy
    b.s[2]    = sz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _create_system(body_t *b,
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
                         long double[:] sz) nogil:

    cdef unsigned int i

    for i in range(n):
        _create_body(&b[i], mass[i], x[i], y[i], z[i], px[i], py[i], pz[i], sx[i], sy[i], sz[i])

    return
    
