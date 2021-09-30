cdef struct body:
    double mass
    double q[3]
    double p[3]
    double s[3]

ctypedef body body_t

cdef struct system:
    body_t *bodies
    unsigned int n

ctypedef system system_t

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
                       double sz) nogil

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
                         double[:] sz) nogil
