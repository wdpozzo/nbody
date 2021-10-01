cdef struct body:
    long double mass
    long double q[3]
    long double p[3]
    long double s[3]

ctypedef body body_t

cdef struct system:
    body_t *bodies
    unsigned int n

ctypedef system system_t

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
                       long double sz) nogil

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
                         long double[:] sz) nogil
