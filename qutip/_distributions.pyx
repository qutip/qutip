cimport cython
from cython cimport double, complex
cimport numpy as np
import numpy as np
from libc.math cimport pi 
from cmath import exp as cexp, sqrt as csqrt, pi as cpi

@cython.nogil
@cython.cfunc
@cython.locals(x_size=np.npy_intp, j=int, i=int, k=int, temp1=complex, temp2=complex)
@cython.boundscheck(False)
cpdef np.ndarray[np.complex128_t, ndim=1] psi_n_single_fock_multiple_position_complex(int n, np.ndarray[np.complex128_t, ndim=1] x):

    """
    Compute the wavefunction to a complex vector x using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : np.ndarray[np.complex128_t]
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.complex128_t]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position_complex(0, np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([ 0.40583486-0.63205035j, -0.49096842+0.56845369j])
    >>> psi_n_single_fock_multiple_position_complex(61, np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2] result = np.zeros((n + 1, x_size), dtype=np.complex128)
    pi_025 = pi ** (-0.25)

    for j in range(x_size):
        result[0, j] = pi_025 * cexp(-(x[j] ** 2) / 2)

    for i in range(n):
        temp1 = csqrt(2 * (i + 1))
        temp2 = csqrt(i / (i + 1))
        if(i == 0):
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1)
        else:
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1) - temp2 * result[i - 1, k]

    return result[-1, :]


