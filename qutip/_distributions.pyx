cimport cython
from cython cimport double, complex
cimport numpy as np
import numpy as np
from libc.math cimport pi
cdef extern from "<complex>" namespace "std" nogil:
    double complex cexp "exp" (double complex x)
    double complex csqrt "sqrt" (double complex x)

@cython.locals(x_size=np.npy_intp, j=int, i=int, k=int, temp1=complex, temp2=complex)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray psi_fock_multiple_position_complex(int n_max, double complex[::1] x):

    """
    Compute the Fock-state wavefunctions psi_0 ... psi_{n_max} at complex
    positions x using an adapted recurrence relation.

    Parameters
    ----------
    n_max : int
        Highest Fock state number.
    x : np.ndarray[np.complex128_t]
        C-contiguous position(s) at which to evaluate the wavefunctions.

    Returns
    -------
    np.ndarray[np.complex128_t]
        Array of shape ``(n_max + 1, len(x))`` whose row ``n`` is psi_n(x).

    Examples
    --------
    ```python
    >>> psi_fock_multiple_position_complex(1, np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
           [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
    >>> psi_fock_multiple_position_complex(61, np.array([1.0 + 1.0j, 2.0 + 2.0j]))[-1]
    array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    cdef double complex[:, ::1] result = np.zeros((n_max + 1, x_size), dtype=np.complex128)
    cdef double pi_025 = pi ** (-0.25)

    for j in range(x_size):
        result[0, j] = pi_025 * cexp(-(x[j] * x[j]) / 2)

    for i in range(n_max):
        temp1 = csqrt(2 * (i + 1))
        temp2 = csqrt(i / (i + 1))
        if(i == 0):
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1)
        else:
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1) - temp2 * result[i - 1, k]

    return np.asarray(result)


