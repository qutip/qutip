#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
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
    """
        Computes the value of the nested integrals for a given array of
        effective omegas. See eq. (7) in Ref.

        Parameters
        ----------
        ws : double[:]
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

        dt : double
            The time increment.

        a_tol : double, default = 1e-10
            The absolute tolerance used.

        Returns
        -------
        value : complex
            The value of the nested integrals.

        Notes
        -----
        Integrals are done analytically from right to left with integration
        by parts.

    """
    return _compute_integrals(ws, 0, 0., dt, a_tol)


# Recursion helpers work on the effective omega array
# [ws[s] + carry, ws[s + 1], ..., ws[-1]], so no sub-array is ever copied.
cdef complex _compute_integrals(double[:] ws, Py_ssize_t s, double carry, double dt, double a_tol) noexcept nogil:
    cdef double w0 = ws[s] + carry
    if s == ws.shape[0] - 1:
        if abs(w0) < a_tol:
            return dt
        return (-1.j / w0) * (exp(1j * w0 * dt) - 1.)

    if abs(w0) < a_tol:
        return cy_compute_tn_integrals(ws, s + 1, 0., 1, dt, a_tol)

    return (-1j / w0) * (
        _compute_integrals(ws, s + 1, w0, dt, a_tol)
        - _compute_integrals(ws, s + 1, 0., dt, a_tol)
    )


cdef complex cy_compute_tn_integrals(double[:] ws, Py_ssize_t s, double carry, int n, double dt, double a_tol) noexcept nogil:
    """
    Helper function to compute nested integrals when the function to
    integrate is t^n/factorial(n) * exp(1j*omega*t). This happens when
    some effective omegas are 0. In that case, the recursion differs a
    bit from _compute_integrals(). See eq. (7) in Ref.

    Note: Integrals are done analytically from right to left with integration
    by parts.
    """
    cdef complex factor, term1, term2
    cdef double w0
    cdef int j

    if n == 0:
        return _compute_integrals(ws, s, carry, dt, a_tol)

    if n == 20:
        # Max supported n, order of 1e-18
        return 0.

    w0 = ws[s] + carry
    if s == ws.shape[0] - 1:
        if abs(w0) < a_tol:
            return (dt ** (n + 1)) * inv_factorial[n + 1]
        else:
            factor = (-1j/w0) * exp(1j*w0*dt)
            term1 = 0
            for j in range(n+1):
                term1 += ((1j/w0)**j) * (dt**(n-j) * inv_factorial[n-j])
            term2 = (1j / w0)**(n+1)
            return factor * term1 + term2
    else:
        if abs(w0) < a_tol:
            return cy_compute_tn_integrals(ws, s + 1, 0., n + 1, dt, a_tol)
        else:
            factor = -1j / w0
            term1 = cy_compute_tn_integrals(ws, s + 1, w0, n, dt, a_tol)
            term2 = cy_compute_tn_integrals(ws, s, carry, n - 1, dt, a_tol)
            return factor * (term1 - term2)