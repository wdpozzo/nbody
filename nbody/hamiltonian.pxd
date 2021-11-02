import numpy as np
cimport numpy as np
from nbody.body cimport body_t

cdef long double G = 1

cdef long double _modulus(long double x, long double y, long double z) nogil

cdef long double _dot(long double *v1, long double *v2) nogil

cdef long double _hamiltonian(body_t *s, unsigned int N, int order) nogil

cdef long double _kinetic_energy(body_t b) nogil

cdef long double _potential(body_t *s, unsigned int N, int order) nogil

cdef long double _potential_0pn(body_t b1, body_t b2) nogil

cdef long double _potential_1pn(body_t b1, body_t b2) nogil

cdef long double _potential_2pn(body_t b1, body_t b2) nogil

cdef void _gradients(long double **out, body_t *bodies, unsigned int N, int order) nogil

cdef void _gradient(long double *out, body_t b1, body_t b2, int order) nogil

cdef void _gradient_free_particle(long double *out, body_t b1) nogil

cdef void _gradient_0pn(long double *out, body_t b1, body_t b2) nogil

cdef void _gradient_1pn(long double *out, body_t b1, body_t b2) nogil

cdef void _gradient_2pn(long double *out, body_t b1, body_t b2) nogil
