# BSD 3-Clause License
#
# Copyright (c) 2024, Matheus Gomes Cordeiro 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


cimport cython
from cython cimport double, complex
cimport numpy as np
import numpy as np
from libc.math cimport exp, sqrt, pi
from cmath import exp as cexp, sqrt as csqrt, pi as cpi

@cython.nogil
@cython.cfunc
@cython.locals(index=int, r0=double, r1=double, r2=double)
@cython.boundscheck(False)
cpdef double psi_n_single_fock_single_position(int n, double x):

    """
    Compute the wavefunction to a real scalar x using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : double
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        double
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position(0, 1.0)
    0.45558067201133257
    >>> psi_n_single_fock_single_position(61, 1.0)
    -0.2393049199171131
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    cdef np.ndarray[np.float64_t, ndim=1] n_coeffs

    r0 = 0.0
    r1 = (pi ** (-0.25)) * exp(-(x ** 2) / 2)

    for index in range(n):
        if index == 0:
            r2 = 2 * x * (r1 / sqrt(2 * (index + 1)))
            r0 = r1
            r1 = r2
        else:
            r2 = 2 * x * (r1 / sqrt(2 * (index + 1))) - sqrt(index / (index + 1)) * r0 
            r0 = r1
            r1 = r2

    return r1

@cython.nogil
@cython.cfunc
@cython.locals(index=int, r0=complex, r1=complex, r2=complex)
@cython.boundscheck(False)
cpdef double complex psi_n_single_fock_single_position_complex(int n, double complex x):

    """
    Compute the wavefunction to a complex scalar x using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : double
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        double
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position_complex(0,1.0+2.0j)
    (-1.4008797330262455-3.0609780602975003j)
    >>> psi_n_single_fock_single_position_complex(61,1.0+2.0j)
    (-511062135.47555304+131445997.75753704j)
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    cdef np.ndarray[np.float64_t, ndim=1] n_coeffs

    r0 = 0.0 + 0.0j
    r1 = (cpi ** (-0.25)) * cexp(-(x ** 2) / 2)

    for index in range(n):
        if index == 0:
            r2 = 2 * x * (r1 / csqrt(2 * (index + 1)))
            r0 = r1
            r1 = r2
        else:
            r2 = 2 * x * (r1 / csqrt(2 * (index + 1))) - csqrt(index / (index + 1)) * r0 
            r0 = r1
            r1 = r2

    return r1

@cython.nogil
@cython.cfunc
@cython.locals(x_size=np.npy_intp, j=int, i=int, k=int, temp1=double, temp2=double)
@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] psi_n_single_fock_multiple_position(int n, np.ndarray[np.float64_t, ndim=1] x):

    """
    Compute the wavefunction to a real vector x using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : np.ndarray[np.float64_t]
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.float64_t]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position(0, np.array([1.0, 2.0]))
    array([0.45558067, 0.10165379])
    >>> psi_n_single_fock_multiple_position(61, np.array([1.0, 2.0]))
    array([-0.23930492, -0.01677378])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n + 1, x_size), dtype=np.float64)

    for j in range(x_size):
        result[0, j] = (pi ** (-0.25)) * exp(-(x[j] ** 2) / 2)

    for i in range(n):
        temp1 = sqrt(2 * (i + 1))
        temp2 = sqrt(i / (i + 1))
        if(i == 0):
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1)
        else:
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1) - temp2 * result[i - 1, k]

    return result[-1, :]

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

    for j in range(x_size):
        result[0, j] = (cpi ** (-0.25)) * cexp(-(x[j] ** 2) / 2)

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


@cython.nogil
@cython.cfunc
@cython.locals(index=int)
@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] psi_n_multiple_fock_single_position(int n, double x):

    """
    Compute the wavefunction to a real scalar x to all fock states until n using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : double
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.float64_t]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_single_position(0, 1.0)
    array([0.45558067, 0.64428837])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros((n + 1), dtype=np.float64)
    result[0] = (pi ** (-0.25)) * exp(-(x ** 2) / 2)

    for index in range(n):
        if(index == 0):
            result[index + 1] = 2 * x * (result[index] / sqrt(2 * (index + 1)))
        else:
            result[index + 1] = 2 * x * (result[index] / sqrt(2 * (index + 1))) - sqrt(index / (index + 1)) * result[index - 1]

    return result

@cython.nogil
@cython.cfunc
@cython.locals(index=int)
@cython.boundscheck(False)
cpdef np.ndarray[np.complex128_t, ndim=1] psi_n_multiple_fock_single_position_complex(int n, double complex x):

    """
    Compute the wavefunction to a complex scalar x to all fock states until n using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : double complex
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.complex128_t]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_single_position_complex(0, 1.0 +2.0j)
    array([-1.40087973-3.06097806j,  6.67661026-8.29116292j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    cdef np.ndarray[np.complex128_t, ndim=1] result = np.zeros((n + 1), dtype=np.complex128)
    result[0] = (cpi ** (-0.25)) * cexp(-(x ** 2) / 2)

    for index in range(n):
        if(index == 0):
            result[index + 1] = 2 * x * (result[index] / csqrt(2 * (index + 1)))
        else:
            result[index + 1] = 2 * x * (result[index] / csqrt(2 * (index + 1))) - csqrt(index / (index + 1)) * result[index - 1]

    return result

@cython.nogil
@cython.cfunc
@cython.locals(x_size=np.npy_intp, j=int, i=int, k=int, temp1=double, temp2=double)
@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=2] psi_n_multiple_fock_multiple_position(int n, np.ndarray[np.float64_t, ndim=1] x):

    """
    Compute the wavefunction to a real vector x to all fock states until n using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : np.ndarray[np.float64_t]
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.ndarray[np.float64_t]]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_multiple_position(1, np.array([1.0, 2.0]))
    array([[0.45558067, 0.10165379],
           [0.64428837, 0.28752033]])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n + 1, x_size), dtype=np.float64)

    for j in range(x_size):
        result[0, j] = (pi ** (-0.25)) * exp(-(x[j] ** 2) / 2)

    for i in range(n):
        temp1 = sqrt(2 * (i + 1))
        temp2 = sqrt(i / (i + 1))
        if(i == 0):
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1)
        else:
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1) - temp2 * result[i - 1, k]

    return result

@cython.nogil
@cython.cfunc
@cython.locals(x_size=np.npy_intp, j=int, i=int, k=int, temp1=complex, temp2=complex)
@cython.boundscheck(False)
cpdef np.ndarray[np.complex128_t, ndim=2] psi_n_multiple_fock_multiple_position_complex(int n, np.ndarray[np.complex128_t, ndim=1] x):

    """
    Compute the wavefunction to a complex vector x to all fock states until n using adapted recurrence relation.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : np.ndarray[np.complex128_t]
        Position(s) at which to evaluate the wavefunction.


    Returns
    -------
        np.ndarray[np.ndarray[np.complex128_t]]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_multiple_position_complex(1,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
           [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2] result = np.zeros((n + 1, x_size), dtype=np.complex128)

    for j in range(x_size):
        result[0, j] = (cpi ** (-0.25)) * cexp(-(x[j] ** 2) / 2)

    for i in range(n):
        temp1 = csqrt(2 * (i + 1))
        temp2 = csqrt(i / (i + 1))
        if(i == 0):
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1)
        else:
            for k in range(x_size):
                result[i + 1, k] = 2 * x[k] * (result[i, k] / temp1) - temp2 * result[i - 1, k]

    return result