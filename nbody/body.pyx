cimport cython
from libc.stdlib cimport malloc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _create_body(body_t *b,
                       double mass,
                       double x,
                       double y,
                       double z,
                       double px,
                       double py,
                       double pz,
                       double sx,
                       double sy,
                       double sz) nogil:
    
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
                         double[:] mass,
                         double[:] x,
                         double[:] y,
                         double[:] z,
                         double[:] px,
                         double[:] py,
                         double[:] pz,
                         double[:] sx,
                         double[:] sy,
                         double[:] sz) nogil:

    cdef unsigned int i

    for i in range(n):
        _create_body(&b[i], mass[i], x[i], y[i], z[i], px[i], py[i], pz[i], sx[i], sy[i], sz[i])

    return
    
