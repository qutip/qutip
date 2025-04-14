#cython: language_level=3
import numpy as np
cimport cython

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

np_fact = np.zeros(21, dtype=float)
cdef double[:] inv_factorial = np_fact
inv_factorial[0] = 1
for i in range(1, 21):
    inv_factorial[i] = inv_factorial[i-1] / i


cpdef complex cy_compute_integrals(double[:] ws, double dt, double a_tol=1e-10):
    cdef double[:] ws_prime
    if len(ws) == 0:
        return 1.
    elif len(ws) == 1:
        if abs(ws[0]) < a_tol:
            return dt
        else:
            return (-1.j / ws[0]) * (exp(1j * ws[0] * dt) - 1.)
    else:
        if abs(ws[0]) < a_tol:
            return cy_compute_tn_integrals(ws[1:], 1, dt)
        else:
            ws_prime = ws[1:].copy()
            ws_prime[0] += ws[0]
            return (-1j / ws[0]) * (
                cy_compute_integrals(ws_prime, dt)
                - cy_compute_integrals(ws[1:], dt)
            )


cdef complex cy_compute_tn_integrals(double[:] ws, int n, double dt, double a_tol=1e-10):
    cdef complex factor, term1, term2
    cdef double[:] ws_prime
    cdef int j

    if n == 0:
        return cy_compute_integrals(ws, dt)

    if n == 20:
        # Max supported n, order of 1e-18
        return 0.

    if len(ws) == 1:
        if abs(ws[0]) < a_tol:
            return (dt ** (n + 1)) * inv_factorial[n + 1]
        else:
            factor = (-1j/ws[0]) * exp(1j*ws[0]*dt)
            term1 = 0
            for j in range(n+1):
                term1 += ((1j/ws[0])**j) * (dt**(n-j) * inv_factorial[n-j])
            term2 = (1j / ws[0])**(n+1)
            return factor * term1 + term2
    else:
        if abs(ws[0]) < a_tol:
            return cy_compute_tn_integrals(ws[1:], n + 1, dt)
        else:
            factor = -1j / ws[0]
            ws_prime = ws[1:].copy()
            ws_prime[0] += ws[0]
            term1 = cy_compute_tn_integrals(ws_prime, n, dt)
            term2 = cy_compute_tn_integrals(ws, n - 1, dt)
            return factor * (term1 - term2)