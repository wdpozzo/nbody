import numpy as np
cimport numpy as np
from nbody.body cimport body_t

cdef long double G = 1

cdef long double _modulus(long double x, long double y, long double z) nogil
cdef long double _dot(long double *v1, long double *v2) nogil
cdef long double _hamiltonian(body_t *s, unsigned int N, int order)
cdef long double _kinetic_energy(body_t b) nogil
cdef long double _potential(body_t *s, unsigned int N, int order) nogil
cdef long double _potential_0pn(body_t b1, body_t b2) nogil
cdef np.ndarray[long double, mode="c", ndim=2] _gradients(body_t *s, unsigned int N, int order)
cdef long double[:] _gradient(body_t b1, body_t b2, int order)
cdef long double[:] _gradient_0pn(body_t b1, body_t b2)
cdef long double[:] _gradient_1pn(body_t b1, body_t b2)
