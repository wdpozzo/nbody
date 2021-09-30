import numpy as np
cimport numpy as np
from nbody.body cimport body_t

cdef double G = 1

cdef double _modulus(double x, double y, double z) nogil
cdef double _dot(double *v1, double *v2) nogil
cdef double _hamiltonian(body_t *s, unsigned int N, int order)
cdef double _kinetic_energy(body_t b) nogil
cdef double _potential(body_t *s, unsigned int N, int order) nogil
cdef double _potential_0pn(body_t b1, body_t b2) nogil
cdef np.ndarray[double, mode="c", ndim=2] _gradients(body_t *s, unsigned int N, int order)
cdef double[:] _gradient(body_t b1, body_t b2, int order)
cdef double[:] _gradient_0pn(body_t b1, body_t b2)
cdef double[:] _gradient_1pn(body_t b1, body_t b2)
