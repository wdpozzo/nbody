cdef struct body:
    long double mass
    long double q[3]
    long double p[3]
    long double s[3]

ctypedef body body_t

cdef struct fit_coefficients:
    long double m
    long double eta
    long double eta2
    long double eta3
    long double eta4
    long double Stot
    long double Shat
    long double Shat2
    long double Shat3
    long double Shat4
    long double chidiff
    long double chidiff2
    long double sqrt2
    long double sqrt3
    long double sqrt1m4eta

ctypedef fit_coefficients fit_coefficients_t

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

cdef (int, int) _find_mergers(body_t *b, unsigned int nbodies) nogil
cdef int merger(body_t b1, body_t b2, double r) nogil
cdef void _merge_bodies(body_t *b, int i1, int i, unsigned int n) nogil

cdef void _fits_setup(fit_coefficients_t *out, long double m1, long double m2, long double chi1, long double chi2) nogil
cdef double _final_mass(long double m1, long double m2, long double chi1, long double chi2, unsigned int version) nogil
cdef long double _final_spin(long double m1, long double m2, long double chi1, long double chi2, unsigned int version) nogil
