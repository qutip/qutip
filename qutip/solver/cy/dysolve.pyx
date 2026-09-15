#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport cython
from libc.math cimport fabs as abs

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

np_fact = np.zeros(21, dtype=float)
cdef double[:] inv_factorial = np_fact
inv_factorial[0] = 1
for i in range(1, 21):
    inv_factorial[i] = inv_factorial[i-1] / i


cpdef complex cy_compute_integrals(
    double[:] ws, double dt, double a_tol=1e-10
) noexcept nogil:
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
    return cy_compute_core(ws[1:], dt, a_tol, ws[0])


cdef complex cy_compute_core(
    double[:] ws, double dt, double a_tol, double w0
) noexcept nogil:
    if ws.shape[0] == 0:
        if abs(w0) < a_tol:
            return dt
        else:
            return (-1.j / w0) * (exp(1j * w0 * dt) - 1.)
    else:
        if abs(w0) < a_tol:
            return cy_compute_tn_integrals(ws, 1, dt, a_tol, 0.)
        else:
            return (-1j / w0) * (
                cy_compute_core(ws[1:], dt, a_tol, w0 + ws[0])
                - cy_compute_core(ws[1:], dt, a_tol, ws[0])
            )


cdef complex cy_compute_tn_integrals(
    double[:] ws, int n, double dt, double a_tol, double _cum_shift
) noexcept nogil:
    """
        Helper function to compute nested integrals when the function to
        integrate is t^n/factorial(n) * exp(1j*omega*t). This happens when
        some effective omegas are 0. In that case, the recursion differs a
        bit from _compute_integrals().
    """
    cdef complex factor, term1, term2, inv, acc
    cdef int j

    if n == 0:
        return cy_compute_core(ws[1:], dt, a_tol, ws[0] + _cum_shift)

    if n == 20:
        # Max supported n, order of 1e-18
        return 0.

    cdef double w0 = ws[0] + _cum_shift

    if ws.shape[0] == 1:
        if abs(w0) < a_tol:
            return (dt ** (n + 1)) * inv_factorial[n + 1]
        else:
            inv = (1j / w0)
            acc = 1.
            factor = -inv * exp(1j * w0 * dt)
            term1 = 0
            for j in range(n + 1):
                term1 += acc * (dt**(n-j) * inv_factorial[n-j])
                acc *= inv
            term2 = acc
            return factor * term1 + term2
    else:
        if abs(w0) < a_tol:
            return cy_compute_tn_integrals(ws[1:], n + 1, dt, a_tol, 0.)
        else:
            factor = -1j / w0
            term1 = cy_compute_tn_integrals(ws[1:], n, dt, a_tol, w0)
            term2 = cy_compute_tn_integrals(ws, n - 1, dt, a_tol, _cum_shift)
            return factor * (term1 - term2)
