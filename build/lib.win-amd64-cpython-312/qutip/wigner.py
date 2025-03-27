__all__ = [
    'wigner', 'qfunc', 'QFunc', 'spin_q_function', 'spin_wigner',
    'wigner_transform',
]

import numpy as np
import warnings
from numpy import (
    zeros, array, arange, exp, real, conj, pi, copy, sqrt, meshgrid, cos, sin,
)
import scipy.sparse as sp
import scipy.fftpack as ft
import scipy.linalg as la
import scipy.special
from scipy.special import genlaguerre, binom, factorial

try:
    from scipy.special import sph_harm_y
except ImportError:
    from scipy.special import sph_harm

    def sph_harm_y(n, m, polar, azimuthal):
        # sph_harm is set for removal.
        # Same function, but changed parameter order
        return sph_harm(m, n, azimuthal, polar)

import qutip
from qutip import Qobj, ket2dm, jmat
from .solver.parallel import parallel_map
from .utilities import clebsch
from .core import data as _data
from .core.data.eigen import eigh


def wigner_transform(psi, j, fullparity, steps, slicearray):
    """takes the density matrix or state vector of any finite state and
    generates the Wigner function for that state on a sphere, generating a spin
    Wigner function useful for displaying the quasi-probability for a qubit or
    any qudit. For the standard, continuous-variable Wigner function for
    position and momentum variables, wigner() should be used.

    Parameters
    ----------
        psi : qobj
              a state vector or density matrix.
        j : int
            the total angular momentum of the quantum state.
        fullparity : bool
                     should the parity of the full SU space be used?
        steps : int
                number of points at which the Wigner transform is calculated.
        slicearray : list of str
                     the angle slice to be used for each particle in case of a
                     multi-particle quantum state. 'l' yields an equal angle
                     slice. 'x', 'y' and 'z' angle slices can also be chosen.

    Returns
    ----------
        wigner : list of float
                 the wigner transformation at `steps` different theta and phi.

    Raises
    ------
    ComplexWarning
        This can be ignored as it is caused due to rounding errors.

    Notes
    ------
    See example notebook wigner_visualisation.

    References
    ------
    [1] T. Tilma, M. J. Everitt, J. H. Samson, W. J. Munro,
        and K. Nemoto, Phys. Rev. Lett. 117, 180401 (2016).
    [2] R. P. Rundle, P. W. Mills, T. Tilma, J. H. Samson, and
        M. J. Everitt, Phys. Rev. A 96, 022117 (2017).
    """
    if not (psi.type == 'ket' or psi.type == 'operator' or psi.type == 'bra'):
        raise TypeError('Input state is not a valid operator.')

    if psi.isket or psi.isbra:
        rho = psi.proj()
    else:
        rho = psi

    sun = 2   # The order of the SU group

    # calculate total number of particles in quantum state:
    N = np.int32(np.log(np.shape(rho)[0]) / np.log(2 * j + 1))

    theta = np.zeros((N, steps))
    phi = np.zeros((N, steps))

    for i in range(N):
        theta[i, :] = np.linspace(0, np.pi, steps)
        phi[i, :] = np.linspace(0, 2 * np.pi, steps)

    theta, phi = _angle_slice(np.array(slicearray, dtype=str), theta, phi)

    wigner = np.zeros((steps, steps))
    if fullparity:
        pari = _parity(sun**N, j)
    else:
        pari = _parity(sun, j)
    for t in range(steps):
        for p in range(steps):
            kernel = _kernelsu2(theta[:, t], phi[:, p], N, j, pari, fullparity)
            kernel = _data.dense.fast_from_numpy(kernel)
            wigner[t, p] = _data.expect(rho.data, kernel).real
    return wigner


def _parity(N, j):
    """Private function to calculate the parity of the quantum system.
    """
    if j == 0.5:
        pi = np.identity(N) - np.sqrt((N - 1) * N * (N + 1) / 2) * _lambda_f(N)
        return pi / N
    elif j > 0.5:
        mult = np.int32(2 * j + 1)
        matrix = np.zeros((mult, mult))
        foo = np.ones(mult)
        for n in np.arange(-j, j + 1, 1):
            for l in np.arange(0, mult, 1):
                foo[l] = (2 * l + 1) * clebsch(j, l, j, n, 0, n)
            matrix[np.int32(n + j), np.int32(n + j)] = np.sum(foo)
        return matrix / mult


def _lambda_f(N):
    """Private function needed for the calculation of the parity.
    """
    matrix = np.sqrt(2 / (N * (N - 1))) * np.identity(N)
    matrix[-1, -1] = - np.sqrt(2 * (N - 1) / N)
    return matrix


def _kernelsu2(theta, phi, N, j, parity, fullparity):
    """Private function that calculates the kernel for the SU2 unitary group.
    """
    U = np.ones(1)
    # calculate the total rotation matrix (tensor product for each particle):
    for i in range(0, N):
        U = np.kron(U, _rotation_matrix(theta[i], phi[i], j))
    if not fullparity:
        op_parity = parity   # The parity for a one particle system
        for i in range(1, N):
            parity = np.kron(parity, op_parity)
    matrix = U @ parity @ U.conj().T
    return matrix


def _rotation_matrix(theta, phi, j):
    """Private function to calculate the rotation operator for the SU2 kernel.
    """
    return la.expm(1j * phi * jmat(j, 'z').full()) @ \
           la.expm(1j * theta * jmat(j, 'y').full())


def _angle_slice(slicearray, theta, phi):
    """Private function to modify theta and phi for angle slicing.
    """
    xind = np.where(slicearray == 'x')
    theta[xind, :] = np.pi - theta[xind, :]
    phi[xind, :] = -phi[xind, :]
    yind = np.where(slicearray == 'y')
    theta[yind, :] = np.pi - theta[yind, :]
    phi[yind, :] = np.pi - phi[yind, :]
    zind = np.where(slicearray == 'z')
    phi[zind, :] = phi[zind, :] + np.pi
    return theta, phi


def wigner(psi, xvec, yvec=None, method='clenshaw', g=sqrt(2),
           sparse=False, parfor=False):
    """Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.

    Parameters
    ----------

    state : qobj
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    g : float, default: sqrt(2)
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.
        The value of `g` is related to the value of `hbar` in the commutation
        relation `[x, y] = i * hbar` via `hbar=2/g^2` giving the default
        value `hbar=1`.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'}, default: 'clenshaw'
        Select method 'clenshaw' 'iterative', 'laguerre', or 'fft', where 'clenshaw'
        and 'iterative' use an iterative method to evaluate the Wigner functions for density
        matrices :math:`|m><n|`, while 'laguerre' uses the Laguerre polynomials
        in scipy for the same task. The 'fft' method evaluates the Fourier
        transform of the density matrix. The 'iterative' method is default, and
        in general recommended, but the 'laguerre' method is more efficient for
        very sparse density matrices (e.g., superpositions of Fock states in a
        large Hilbert space). The 'clenshaw' method is the preferred method for
        dealing with density matrices that have a large number of excitations
        (>~50). 'clenshaw' is a fast and numerically stable method.

    sparse : bool, optional
        Tells the default solver whether or not to keep the input density
        matrix in sparse format.  As the dimensions of the density matrix
        grow, setthing this flag can result in increased performance.

    parfor : bool, optional
        Flag for calculating the Laguerre polynomial based Wigner function
        method='laguerre' in parallel using the parfor function.


    Returns
    -------

    W : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].

    yvex : array
        FFT ONLY. Returns the y-coordinate values calculated via the Fourier
        transform.

    Notes
    -----
    The 'fft' method accepts only an xvec input for the x-coordinate.
    The y-coordinates are calculated internally.

    References
    ----------

    Ulf Leonhardt,
    Measuring the Quantum State of Light, (Cambridge University Press, 1997)

    """

    if not (psi.type == 'ket' or psi.type == 'oper' or psi.type == 'bra'):
        raise TypeError('Input state is not a valid operator.')

    if method == 'fft':
        return _wigner_fourier(psi, xvec, g)

    if psi.isket or psi.isbra:
        rho = psi.proj()
    else:
        rho = psi

    if method == 'iterative':
        return _wigner_iterative(rho, xvec, yvec, g)

    elif method == 'laguerre':
        return _wigner_laguerre(rho, xvec, yvec, g, parfor)

    elif method == 'clenshaw':
        return _wigner_clenshaw(rho, xvec, yvec, g, sparse=sparse)

    else:
        raise TypeError(
            "method must be either 'iterative', 'laguerre', or 'fft'.")


def _wigner_iterative(rho, xvec, yvec, g=sqrt(2)):
    r"""
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.

    The Wigner function is calculated as
    :math:`W = \sum_{mn} \rho_{mn} W_{mn}` where :math:`W_{mn}` is the Wigner
    function for the density matrix :math:`|m><n|`.

    In this implementation, for each row m, Wlist contains the Wigner functions
    Wlist = [0, ..., W_mm, ..., W_mN]. As soon as one W_mn Wigner function is
    calculated, the corresponding contribution is added to the total Wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """

    M = np.prod(rho.shape[0])
    X, Y = meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)

    Wlist = array([zeros(np.shape(A), dtype=complex) for k in range(M)])
    Wlist[0] = exp(-2.0 * abs(A) ** 2) / pi

    W = real(rho[0, 0]) * real(Wlist[0])
    for n in range(1, M):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / sqrt(n)
        W += 2 * real(rho[0, n] * Wlist[n])

    for m in range(1, M):
        temp = copy(Wlist[m])
        Wlist[m] = (2 * conj(A) * temp - sqrt(m) * Wlist[m - 1]) / sqrt(m)

        # Wlist[m] = Wigner function for |m><m|
        W += real(rho[m, m] * Wlist[m])

        for n in range(m + 1, M):
            temp2 = (2 * A * Wlist[n - 1] - sqrt(m) * temp) / sqrt(n)
            temp = copy(Wlist[n])
            Wlist[n] = temp2

            # Wlist[n] = Wigner function for |m><n|
            W += 2 * real(rho[m, n] * Wlist[n])

    return 0.5 * W * g ** 2


def _wigner_laguerre(rho, xvec, yvec, g, parallel):
    r"""
    Using Laguerre polynomials from scipy to evaluate the Wigner function for
    the density matrices :math:`|m><n|`, :math:`W_{mn}`. The total Wigner
    function is calculated as :math:`W = \sum_{mn} \rho_{mn} W_{mn}`.
    """

    M = np.prod(rho.shape[0])
    X, Y = meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)
    W = zeros(np.shape(A))

    # compute wigner functions for density matrices |m><n| and
    # weight by all the elements in the density matrix
    B = 4 * abs(A) ** 2
    if sp.isspmatrix_csr(rho.data):
        # for compress sparse row matrices
        if parallel:
            iterator = (
                (m, rho, A, B) for m in range(len(rho.data.indptr) - 1))
            W1_out = parallel_map(_par_wig_eval, iterator)
            W += sum(W1_out)
        else:
            for m in range(len(rho.data.indptr) - 1):
                for jj in range(rho.data.indptr[m], rho.data.indptr[m + 1]):
                    n = rho.data.indices[jj]

                    if m == n:
                        W += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))

                    elif n > m:
                        W += 2.0 * real(rho[m, n] * (-1) ** m *
                                        (2 * A) ** (n - m) *
                                        sqrt(factorial(m) / factorial(n)) *
                                        genlaguerre(m, n - m)(B))
    else:
        # for dense density matrices
        B = 4 * abs(A) ** 2
        for m in range(M):
            if abs(rho[m, m]) > 0.0:
                W += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))
            for n in range(m + 1, M):
                if abs(rho[m, n]) > 0.0:
                    W += 2.0 * real(rho[m, n] * (-1) ** m *
                                    (2 * A) ** (n - m) *
                                    sqrt(factorial(m) / factorial(n)) *
                                    genlaguerre(m, n - m)(B))

    return 0.5 * W * g ** 2 * np.exp(-B / 2) / pi


def _par_wig_eval(args):
    """
    Private function for calculating terms of Laguerre Wigner function
    using parfor.
    """
    m, rho, A, B = args
    W1 = zeros(np.shape(A))
    for jj in range(rho.data.indptr[m], rho.data.indptr[m + 1]):
        n = rho.data.indices[jj]

        if m == n:
            W1 += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))

        elif n > m:
            W1 += 2.0 * real(rho[m, n] * (-1) ** m *
                             (2 * A) ** (n - m) *
                             sqrt(factorial(m) / factorial(n)) *
                             genlaguerre(m, n - m)(B))
    return W1


def _wigner_fourier(psi, xvec, g=np.sqrt(2)):
    """
    Evaluate the Wigner function via the Fourier transform.
    """
    if psi.type == 'bra':
        psi = psi.dag()
    if psi.type == 'ket':
        return _psi_wigner_fft(psi.full(), xvec, g)
    elif psi.type == 'oper':
        eig_vals, eig_vecs = eigh(psi.full())
        W = 0
        for ii in range(psi.shape[0]):
            W1, yvec = _psi_wigner_fft(
                np.reshape(eig_vecs[:, ii], (psi.shape[0], 1)), xvec, g)
            W += eig_vals[ii] * W1
        return W, yvec


def _psi_wigner_fft(psi, xvec, g=sqrt(2)):
    """
    FFT method for a single state vector.  Called multiple times when the
    input is a density matrix.
    """
    n = len(psi)
    A = _osc_eigen(n, xvec * g / np.sqrt(2))
    xpsi = np.dot(psi.T, A)
    W, yvec = _wigner_fft(xpsi, xvec * g / np.sqrt(2))
    return (0.5 * g ** 2) * np.real(W.T), yvec * np.sqrt(2) / g


def _wigner_fft(psi, xvec):
    """
    Evaluates the Fourier transformation of a given state vector.
    Returns the corresponding density matrix and range
    """
    n = 2*len(psi.T)
    r1 = np.concatenate(
        (np.array([0]), np.fliplr(psi.conj()).ravel(), np.zeros(n//2 - 1))
    )
    r2 = np.concatenate(
        (np.array([0]), psi.ravel(), np.zeros(n//2 - 1))
    )
    w = la.toeplitz(np.zeros(n//2), r1) * \
        np.flipud(la.toeplitz(np.zeros(n//2), r2))
    w = np.concatenate((w[:, n//2:n], w[:, 0:n//2]), axis=1)
    w = ft.fft(w)
    w = np.real(np.concatenate((w[:, 3*n//4:n+1], w[:, 0:n//4]), axis=1))
    p = np.arange(-n/4, n/4)*np.pi / (n*(xvec[1] - xvec[0]))
    w = w / (p[1] - p[0]) / n
    return w, p


def _osc_eigen(N, pnts):
    """
    Vector of and N-dim oscillator eigenfunctions evaluated
    at the points in pnts.
    """
    pnts = np.asarray(pnts)
    lpnts = len(pnts)
    A = np.zeros((N, lpnts))
    A[0, :] = np.exp(-pnts ** 2 / 2.0) / pi ** 0.25
    if N == 1:
        return A
    else:
        A[1, :] = np.sqrt(2) * pnts * A[0, :]
        for k in range(2, N):
            A[k, :] = np.sqrt(2.0 / k) * pnts * A[k - 1, :] - \
                np.sqrt((k - 1.0) / k) * A[k - 2, :]
        return A


def _wigner_clenshaw(rho, xvec, yvec, g=sqrt(2), sparse=False):
    r"""
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.

    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / \sqrt(L!)` where
    :math:`c_L = \sum_n \rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    """

    M = np.prod(rho.shape[0])
    X,Y = np.meshgrid(xvec, yvec)
    #A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (X + 1.0j * Y) #this is A2 = 2*A

    B = np.abs(A2)
    B *= B
    w0 = (2*rho[0, -1])*np.ones_like(A2)
    L = M-1
    #calculation of \sum_{L} c_L (2x)^L / \sqrt(L!)
    #using Horner's method
    if not sparse:
        rho = rho.full() * (2*np.ones((M,M)) - np.diag(np.ones(M)))
        while L > 0:
            L -= 1
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    else:
        # TODO: fix dispatch.
        _rho = _data.to(_data.CSR, rho.data).as_scipy()
        while L > 0:
            L -= 1
            diag = _rho.diagonal(L)
            if L != 0:
                diag *= 2
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, diag) + w0 * A2 * (L+1)**-0.5

    return w0.real * np.exp(-B*0.5) * (g*g*0.5 / pi)


def _wig_laguerre_val(L, x, c):
    r"""
    this is evaluation of polynomial series inspired by hermval from numpy.
    Returns polynomial series

    .. math:
        \sum_n b_n LL_n^L,

    where

    .. math:
        LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]

    The evaluation uses Clenshaw recursion.
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5

    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5


# -----------------------------------------------------------------------------
# Q FUNCTION
#
def _qfunc_check_state(state: Qobj):
    if not isinstance(state, Qobj):
        raise TypeError(f"state must be Qobj, but is {state}")
    # This is only approximate, but it's enough for our purposes; doing more
    # than this would take computational effort we don't _need_ to do.
    isdm = (
        state.isoper
        and state.dims[0] == state.dims[1]
        and state.isherm
        and abs(state.tr() - 1) < qutip.settings.core['atol']
    )
    if not (state.isket or isdm):
        raise ValueError(
            f"state must be a ket or density matrix, but is {state}"
        )
    if len(state.dims[0]) != 1:
        raise ValueError(
            "state must not have tensor structure, but has dimensions:"
            f" {state.dims}"
        )
    return state


def _qfunc_check_coordinates(xvec, yvec):
    if np.isscalar(xvec) or xvec is None:
        raise TypeError("xvec must be array-like, but is " + repr(xvec))
    if np.isscalar(yvec) or yvec is None:
        raise TypeError("yvec must be array-like, but is " + repr(yvec))
    xvec = np.asarray(xvec, dtype=np.float64)
    yvec = np.asarray(yvec, dtype=np.float64)
    if xvec.ndim != 1 or yvec.ndim != 1:
        raise ValueError(
            f"xvec and yvec must be 1D, but have shapes {xvec.shape} and {yvec.shape}."
        )
    return xvec, yvec


class _QFuncCoherentGrid:
    """
    Internal function to compute coherent state operators corresponding to a
    grid of complex values in phase space.  For efficiency reasons, this class
    produces the adjoint of the coherent states, to save allocations when
    calculating inner products later.

    Examples
    --------
    Initialise the grid calculator.

    >>> xvec = yvec = np.linspace(-1, 1, 21)
    >>> g = np.sqrt(0.5)
    >>> max_ns = 10
    >>> grid = _QFuncCoherentGrid(xvec, yvec, g)

    The naive construction of the grid is

    >>> xs, ys = np.meshgrid(xvec, yvec)
    >>> all_alphas = 0.5 * g * (xs + 1j*ys)
    >>> naive = np.array([
    ...     [
    ...         qutip.coherent(max_ns, alpha, method='analytic')
    ...             .dag().full().ravel()
    ...         for alpha in x_alphas
    ...     ]
    ...     for y_alphas in all_alphas
    ... ])

    The naive approach is typically several of orders of magnitude slower than
    this class, which uses much simpler vectorised operations.  The outputs are
    within close tolerance, however:

    >>> np.allclose(naive, grid(max_ns))
    True
    >>> np.allclose(naive[:, :, 4:7], grid(4, 7))
    True
    """
    def __init__(self, xvec, yvec, g: float):
        self.xvec, self.yvec = _qfunc_check_coordinates(xvec, yvec)
        x, y = np.meshgrid(0.5 * g * self.xvec, 0.5 * g * self.yvec)
        self.grid = np.empty(x.shape, dtype=np.complex128)
        self.grid.real = x
        # We produce the adjoint of the coherent states to save an operation
        # later when computing dot products, hence the negative imaginary part.
        self.grid.imag = -y
        self.prefactor = np.exp(-0.5 * (x * x + y * y)).astype(np.complex128)

    def _start(self, first: int):
        """
        Get the coherent state matrix corresponding to the first needed Fock
        state.
        """
        if first == 0:
            return self.prefactor.copy()
        out = np.power(self.grid, first)
        out *= self.prefactor
        return out

    def __call__(self, first: int, last: int = None):
        """
        Get a 3D array of shape ``(yvec.size, xvec.size, last - first)`` of the
        coherent-state vectors for all the Fock states in the range ``first``
        to ``last``, excluding the end point.  The first two axes are the y-
        and x-coordinates of phase space (i.e. Cartesian indexing, like
        ``numpy.meshgrid``), and the last runs over the selected range of
        Fock-space dimensions.
        """
        ns = np.arange(first, last).reshape(1, 1, -1)
        # Technically we could avoid hitting the limits of floating-point
        # exponents for longer by doing all this in logarithmic space (using
        # scipy.special.gammaln), but that ends up involving more
        # floating-point operations overall, and needs special care around the
        # point alpha = 0 to avoid nan appearing, due to how Python handles
        # mixed-width arithmetic operations.
        out = np.empty(self.grid.shape + (ns.size,), dtype=np.complex128)
        out[:, :, 0] = self._start(ns.flat[0])
        for i in range(ns.size - 1):
            out[:, :, i+1] = out[:, :, i] * self.grid
        out /= np.sqrt(scipy.special.factorial(ns))
        return out


class QFunc:
    r"""
    Class-based method of calculating the Husimi-Q function of many different
    quantum states at fixed phase-space points ``0.5*g* (xvec + i*yvec)``.
    This class has slightly higher first-usage costs than :obj:`.qfunc`, but
    subsequent operations will be several times faster. However, it can require
    quite a lot of memory. Call the created object as a function to retrieve
    the Husimi-Q function.

    Parameters
    ----------
    xvec, yvec : array_like
        x- and y-coordinates at which to calculate the Husimi-Q function.

    g : float, default: sqrt(2)
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of `hbar` in the commutation relation
        :math:`[x,\,y] = i\hbar` via :math:`\hbar=2/g^2`, so the default
        corresponds to :math:`\hbar=1`.

    memory : real, default: 1024
        Size in MB that may be used internally as workspace.  This class will
        raise ``MemoryError`` if subsequently passed a state of sufficiently
        large dimension that this bound would be exceeded.  In those cases, use
        :obj:`.qfunc` with ``precompute_memory=None`` instead to force using
        the slower, more memory-efficient algorithm.

    Examples
    --------
    Initialise the class for a square set of coordinates, with some states we
    want to investigate.

    >>> xvec = np.linspace(-2, 2, 101)
    >>> states = [qutip.rand_dm(10) for _ in [None]*10]
    >>> qfunc = qutip.QFunc(xvec, xvec)

    Now we can calculate the Husimi-Q function over each of the states more
    efficiently with:

    >>> husimiq = np.array([qfunc(state) for state in states])

    See Also
    --------
    :obj:`.qfunc` :
        A single function version, which will involve computing several
        quantities multiple times in order to use less memory.
    """

    def __init__(
        self, xvec, yvec, g: float = np.sqrt(2), memory: float = 1024
    ):
        self._g = g
        self._coherent_grid = _QFuncCoherentGrid(xvec, yvec, g)
        # 16 bytes per complex, 1024**2 bytes per MB.
        self._size_mb = self._coherent_grid.grid.size * 16 / (1024 ** 2)
        self._memory_mb = memory
        self._max_size = int(self._memory_mb // self._size_mb)
        self._current_size = 0
        self._cache = None

    def _alphas(self, size: int):
        r"""
        Retrive the full tensor of (the conjugate of) coherent states over all
        values of :math:`\alpha`, for states of dimension ``size``.
        """
        if self._current_size >= size:
            return self._cache[:, :, :size]
        if size > self._max_size:
            requirement = self._size_mb * size
            raise MemoryError(
                f"Refusing to precompute up to {size} basis states."
                f" This would require {requirement:.2f} MB,"
                f" but only {self._memory_mb} MB is allowed."
            )
        if self._cache is None:
            self._cache = self._coherent_grid(self._current_size, size)
        else:
            self._cache = np.dstack(
                [self._cache, self._coherent_grid(self._current_size, size)]
            )
        self._current_size = size
        return self._cache

    def _single(self, vector: np.ndarray, alphas: np.ndarray):
        r"""
        Get the Q function (without the :math:`\pi` scaling factor) of a single
        state vector.
        """
        return np.abs(np.dot(alphas, (self._g * 0.5) * vector)) ** 2

    def __call__(self, state: Qobj):
        """
        Get the Husimi-Q function for the given state vector or density matrix,
        over the coordinates used to initialise the class.  If called multiple
        times, the states do not need to have the same dimensions, but none of
        them can have tensor-product structure.
        """
        state = _qfunc_check_state(state)
        alphas = self._alphas(state.shape[0])
        if state.isket:
            return self._single(state.full().ravel(), alphas) / np.pi
        # We don't use Qobj.eigenstates() to avoid building many unnecessary
        # CSR versions of dense matrices.
        values, vectors = eigh(state.full())
        vectors = vectors.T
        out = values[0] * self._single(vectors[0], alphas)
        for value, vector in zip(values[1:], vectors[1:]):
            out += value * self._single(vector, alphas)
        return out / np.pi


def _qfunc_iterative_single(
    vector: np.ndarray, alpha_grid: _QFuncCoherentGrid, g: float,
):
    r"""
    Get the Q function (without the :math:`\pi` scaling factor) of a single
    state vector, using the iterative algorithm which recomputes the powers of
    the coherent-state matrix.
    """
    ns = np.arange(vector.shape[0])
    out = np.polyval(
        (0.5*g * vector / np.sqrt(scipy.special.factorial(ns)))[::-1],
        alpha_grid.grid,
    )
    out *= alpha_grid.prefactor
    return np.abs(out)**2


def qfunc(
    state: Qobj,
    xvec,
    yvec,
    g: float = sqrt(2),
    precompute_memory: float = 1024,
):
    r"""
    Husimi-Q function of a given state vector or density matrix at phase-space
    points ``0.5 * g * (xvec + i*yvec)``.

    Parameters
    ----------
    state : :obj:`.Qobj`
        A state vector or density matrix.  This cannot have tensor-product
        structure.

    xvec, yvec : array_like
        x- and y-coordinates at which to calculate the Husimi-Q function.

    g : float, default: sqrt(2)
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of :math:`\hbar` in the commutation relation
        :math:`[x,\,y] = i\hbar` via :math:`\hbar=2/g^2`, so the default
        corresponds to :math:`\hbar=1`.

    precompute_memory : real, default: 1024
        Size in MB that may be used during calculations as working space when
        dealing with density-matrix inputs.  This is ignored for state-vector
        inputs.  The bound is not quite exact due to other, order-of-magnitude
        smaller, intermediaries being necessary, but is a good approximation.
        If you want to use the same iterative algorithm for density matrices
        that is used for single kets, set ``precompute_memory=None``.

    Returns
    -------
    ndarray
        Values representing the Husimi-Q function calculated over the specified
        range ``[xvec, yvec]``.

    See Also
    --------
    :obj:`.QFunc` :
        a class-based version, more efficient if you want to calculate the
        Husimi-Q function for several states over the same coordinates.
    """
    state = _qfunc_check_state(state)
    xvec, yvec = _qfunc_check_coordinates(xvec, yvec)
    required_memory = state.shape[0] * xvec.size * yvec.size * 16 / (1024 ** 2)
    enough_memory = (
        precompute_memory is not None
        and precompute_memory > required_memory
    )
    if state.isoper and enough_memory:
        return QFunc(xvec, yvec, g)(state)
    if precompute_memory is not None and state.isoper:
        warnings.warn(
            "Falling back to iterative algorithm due to lack of memory."
            f" Needed {required_memory:.2f} MB, but only allowed to use"
            f" {precompute_memory:.2f} MB.  Increase `precompute_memory` to"
            " raise limit, or set to `None` to suppress warning."
        )
    alpha_grid = _QFuncCoherentGrid(xvec, yvec, g)
    if state.isket:
        out = _qfunc_iterative_single(state.full().ravel(), alpha_grid, g)
        out /= np.pi
        return out
    # We don't use Qobj.eigenstates() to avoid building many unnecessary CSR
    # versions of dense matrices.
    values, vectors = eigh(state.full())
    vectors = vectors.T
    out = values[0] * _qfunc_iterative_single(vectors[0], alpha_grid, g)
    for value, vector in zip(values[1:], vectors[1:]):
        out += value * _qfunc_iterative_single(vector, alpha_grid, g)
    out /= np.pi
    return out


# -----------------------------------------------------------------------------
# PSEUDO DISTRIBUTION FUNCTIONS FOR SPINS
#
def spin_q_function(rho, theta, phi):
    r"""The Husimi Q function for spins is defined as ``Q(theta, phi) =
    SCS.dag() * rho * SCS`` for the spin coherent state ``SCS = spin_coherent(
    j, theta, phi)`` where j is the spin length.
    The implementation here is more efficient as it doesn't
    generate all of the SCS at theta and phi (see references).

    The spin Q function is normal when integrated over the surface of the
    sphere

    .. math:: \frac{4 \pi}{2j + 1}\int_\phi \int_\theta
              Q(\theta, \phi) \sin(\theta) d\theta d\phi = 1

    Parameters
    ----------
    state : qobj
        A state vector or density matrix for a spin-j quantum system.
    theta : array_like
        Polar (colatitude) angle at which to calculate the Husimi-Q function.
    phi : array_like
        Azimuthal angle at which to calculate the Husimi-Q function.

    Returns
    -------
    Q, THETA, PHI : 2d-array
        Values representing the spin Husimi Q function at the values specified
        by THETA and PHI.

    References
    ----------
    [1] Lee Loh, Y., & Kim, M. (2015). American J. of Phys., 83(1), 30–35.
    https://doi.org/10.1119/1.4898595

    """

    if rho.isket or rho.isbra:
        rho = rho.proj()

    J = rho.shape[0]
    j = (J - 1) / 2

    THETA, PHI = meshgrid(theta, phi)

    Q = np.zeros_like(THETA, dtype=complex)
    data = rho.full()

    for m1 in arange(-j, j + 1):
        Q += binom(2 * j, j + m1) * cos(THETA / 2) ** (2 * (j + m1)) * \
             sin(THETA / 2) ** (2 * (j - m1)) * \
             data[int(j - m1), int(j - m1)]

        for m2 in arange(m1 + 1, j + 1):
            Q += (sqrt(binom(2 * j, j + m1)) * sqrt(binom(2 * j, j + m2)) *
                  cos(THETA / 2) ** (2 * j + m1 + m2) *
                  sin(THETA / 2) ** (2 * j - m1 - m2)) * \
             (exp(1j * (m1 - m2) * PHI) * data[int(j - m1), int(j - m2)] +
              exp(1j * (m2 - m1) * PHI) * data[int(j - m2), int(j - m1)])

    return Q.real, THETA, PHI


def _rho_kq(rho, j, k, q):
    """
    This calculates the trace of the multipole operator T_kq and the density
    matrix rho for use in the spin Wigner quasiprobability distribution.

    Parameters
    ----------
    rho : qobj
        A density matrix for a spin-j quantum system.
    j : float
        The spin length of the system.
    k : int
        Spherical harmonic degree
    q : int
        Spherical harmonic order

    Returns
    -------
    v : float
        Overlap of state with multipole operator T_kq
    """

    v = 0j
    data = rho.full()
    for m1 in arange(-j, j+1):
        for m2 in arange(-j, j+1):
            v += (
                    (-1) ** (2 * j - k - m1 - m2)
                    * np.sqrt((2 * k + 1) / (2 * j + 1))
                    * clebsch(j, k, j, -m1, q, -m2)
                    * data[int(j - m1), int(j - m2)]
            )
    return v


def spin_wigner(rho, theta, phi):
    r"""Wigner function for a spin-j system.

    The spin W function is normal when integrated over the surface of the
    sphere

    .. math:: \sqrt{\frac{4 \pi}{2j + 1}}\int_\phi \int_\theta
              W(\theta,\phi) \sin(\theta) d\theta d\phi = 1


    Parameters
    ----------
    state : qobj
        A state vector or density matrix for a spin-j quantum system.
    theta : array_like
        Polar (colatitude) angle at which to calculate the W function.
    phi : array_like
        Azimuthal angle at which to calculate the W function.

    Returns
    -------
    W, THETA, PHI : 2d-array
        Values representing the spin Wigner function at the values specified
        by THETA and PHI.

    References
    ----------
    [1] Agarwal, G. S. (1981). Phys. Rev. A, 24(6), 2889–2896.
    https://doi.org/10.1103/PhysRevA.24.2889

    [2] Dowling, J. P., Agarwal, G. S., & Schleich, W. P. (1994).
    Phys. Rev. A, 49(5), 4101–4109. https://doi.org/10.1103/PhysRevA.49.4101

    [3] Conversion between Wigner 3-j symbol and Clebsch-Gordan coefficients
    taken from Wikipedia (https://en.wikipedia.org/wiki/3-j_symbol)

    """
    if rho.isket or rho.isbra:
        rho = rho.proj()

    J = rho.shape[0]
    j = (J - 1) / 2

    THETA, PHI = meshgrid(theta, phi)

    W = np.zeros_like(THETA, dtype=complex)

    for k in range(int(2 * j)+1):
        for q in arange(-k, k+1):
            W += _rho_kq(rho, j, k, q) * sph_harm_y(k, q, THETA, PHI)

    return W.real, THETA, PHI
