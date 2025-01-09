__all__ = ['entropy_vn', 'entropy_linear', 'entropy_mutual', 'negativity',
           'concurrence', 'entropy_conditional', 'entangling_power',
           'entropy_relative']

from .core.numpy_backend import np
from .partial_transpose import partial_transpose
from . import (ptrace, tensor, sigmay, ket2dm,
               expand_operator)
from .core import data as _data


def entropy_vn(rho, base=np.e, sparse=False):
    """
    Von-Neumann entropy of density matrix

    Parameters
    ----------
    rho : qobj
        Density matrix.
    base : {e, 2}, default: e
        Base of logarithm.
    sparse : bool, default: False
        Use sparse eigensolver.

    Returns
    -------
    entropy : float
        Von-Neumann entropy of `rho`.

    Examples
    --------
    >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
    >>> entropy_vn(rho,2)
    1.0

    """
    if rho.type == 'ket' or rho.type == 'bra':
        rho = ket2dm(rho)
    vals = rho.eigenenergies(sparse=sparse)
    threshold = 1e-17
    nzvals = np.where(vals < threshold, threshold, vals)
    if base == 2:
        logvals = np.log2(nzvals)
    elif base == np.e:
        logvals = np.log(nzvals)
    else:
        raise ValueError("Base must be 2 or e.")
    return np.real(-sum(nzvals * logvals))


def entropy_linear(rho):
    """
    Linear entropy of a density matrix.

    Parameters
    ----------
    rho : qobj
        sensity matrix or ket/bra vector.

    Returns
    -------
    entropy : float
        Linear entropy of rho.

    Examples
    --------
    >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
    >>> entropy_linear(rho)
    0.5

    """
    if rho.type == 'ket' or rho.type == 'bra':
        rho = ket2dm(rho)
    return np.real(1.0 - (rho ** 2).tr())


def concurrence(rho):
    """
    Calculate the concurrence entanglement measure for a two-qubit state.

    Parameters
    ----------
    state : qobj
        Ket, bra, or density matrix for a two-qubit state.

    Returns
    -------
    concur : float
        Concurrence

    References
    ----------

    .. [1] `https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)`

    """
    if rho.isket and rho.dims != [[2, 2], [1, 1]]:
        raise Exception("Ket must be tensor product of two qubits.")

    elif rho.isbra and rho.dims != [[1, 1], [2, 2]]:
        raise Exception("Bra must be tensor product of two qubits.")

    elif rho.isoper and rho.dims != [[2, 2], [2, 2]]:
        raise Exception("Density matrix must be tensor product of two qubits.")

    if rho.isket or rho.isbra:
        rho = ket2dm(rho)

    sysy = tensor(sigmay(), sigmay())

    rho_tilde = (rho * sysy) * (rho.conj() * sysy)

    evals = rho_tilde.eigenenergies()

    # abs to avoid problems with sqrt for very small negative numbers
    evals = abs(np.sort(np.real(evals)))

    sqrt_evals = np.sqrt(evals)
    lsum = sqrt_evals[3] - sqrt_evals[2] - sqrt_evals[1] - sqrt_evals[0]
    return np.maximum(0, lsum)


def negativity(rho, subsys, method='tracenorm', logarithmic=False):
    """
    Compute the negativity for a multipartite quantum system described
    by the density matrix rho. The subsys argument is an index that
    indicates which system to compute the negativity for.

    .. note::

        Experimental.
    """
    if rho.isket or rho.isbra:
        rho = ket2dm(rho)
    mask = [idx == subsys for idx, n in enumerate(rho.dims[0])]
    rho_pt = partial_transpose(rho, mask)

    if method == 'tracenorm':
        N = ((rho_pt.dag() * rho_pt).sqrtm().tr().real - 1)/2.0
    elif method == 'eigenvalues':
        l = rho_pt.eigenenergies()
        N = ((abs(l)-l)/2).sum()
    else:
        raise ValueError("Unknown method %s" % method)

# Return the negativity value (or its logarithm if specified)
    if logarithmic:
        return np.log2(2 * N + 1)
    else:
        return N


def entropy_mutual(rho, selA, selB, base=np.e, sparse=False):
    """
    Calculates the mutual information S(A:B) between selection
    components of a system density matrix.

    Parameters
    ----------
    rho : qobj
        Density matrix for composite quantum systems
    selA : int/list
        `int` or `list` of first selected density matrix components.
    selB : int/list
        `int` or `list` of second selected density matrix components.
    base : {e, 2}, default: e
        Base of logarithm.
    sparse : bool, default: False
        Use sparse eigensolver.

    Returns
    -------
    ent_mut : float
       Mutual information between selected components.

    """
    if isinstance(selA, int):
        selA = [selA]
    if isinstance(selB, int):
        selB = [selB]
    if rho.type != 'oper':
        raise TypeError("Input must be a density matrix.")
    if (len(selA) + len(selB)) != len(rho.dims[0]):
        raise TypeError("Number of selected components must match " +
                        "total number.")

    rhoA = ptrace(rho, selA)
    rhoB = ptrace(rho, selB)
    out = (entropy_vn(rhoA, base, sparse=sparse) +
           entropy_vn(rhoB, base, sparse=sparse) -
           entropy_vn(rho, base, sparse=sparse))
    return out


def entropy_relative(rho, sigma, base=np.e, sparse=False, tol=1e-12):
    """
    Calculates the relative entropy S(rho||sigma) between two density
    matrices.

    Parameters
    ----------
    rho : :class:`.Qobj`
        First density matrix (or ket which will be converted to a density
        matrix).
    sigma : :class:`.Qobj`
        Second density matrix (or ket which will be converted to a density
        matrix).
    base : {e, 2}, default: e
        Base of logarithm. Defaults to e.
    sparse : bool, default: False
        Flag to use sparse solver when determining the eigenvectors
        of the density matrices. Defaults to False.
    tol : float, default: 1e-12
        Tolerance to use to detect 0 eigenvalues or dot producted between
        eigenvectors. Defaults to 1e-12.

    Returns
    -------
    rel_ent : float
        Value of relative entropy. Guaranteed to be greater than zero
        and should equal zero only when rho and sigma are identical.

    Examples
    --------

    First we define two density matrices:

    >>> rho = qutip.ket2dm(qutip.ket("00"))
    >>> sigma = rho + qutip.ket2dm(qutip.ket("01"))
    >>> sigma = sigma.unit()

    Then we calculate their relative entropy using base 2 (i.e. ``log2``)
    and base e (i.e. ``log``).

    >>> qutip.entropy_relative(rho, sigma, base=2)
    1.0
    >>> qutip.entropy_relative(rho, sigma)
    0.6931471805599453

    References
    ----------

    See Nielsen & Chuang, "Quantum Computation and Quantum Information",
    Section 11.3.1, pg. 511 for a detailed explanation of quantum relative
    entropy.
    """
    if rho.isket:
        rho = ket2dm(rho)
    if sigma.isket:
        sigma = ket2dm(sigma)
    if not rho.isoper or not sigma.isoper:
        raise TypeError("Inputs must be density matrices.")
    if rho.dims != sigma.dims:
        raise ValueError("Inputs must have the same shape and dims.")
    if base == 2:
        log_base = np.log2
    elif base == np.e:
        log_base = np.log
    else:
        raise ValueError("Base must be 2 or e.")
    # S(rho || sigma) = sum_i(p_i log p_i) - sum_ij(p_i P_ij log q_i)
    #
    # S is +inf if the kernel of sigma (i.e. svecs[svals == 0]) has non-trivial
    # intersection with the support of rho (i.e. rvecs[rvals != 0]).
    rvals, rvecs = _data.eigs(rho.data, rho.isherm, True)
    rvecs = rvecs.to_array().T
    if any(abs(np.imag(rvals)) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = np.real(rvals)
    svals, svecs = _data.eigs(sigma.data, sigma.isherm, True)
    svecs = svecs.to_array().T
    if any(abs(np.imag(svals)) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = np.real(svals)
    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(np.inner(rvecs, np.conj(svecs))) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return np.inf
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    # Calculate S
    S = nzrvals @ log_base(nzrvals) - rvals @ P @ log_base(svals)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return np.maximum(0, S)


def entropy_conditional(rho, selB, base=np.e, sparse=False):
    """
    Calculates the conditional entropy :math:`S(A|B)=S(A,B)-S(B)`
    of a selected density matrix component.

    Parameters
    ----------
    rho : qobj
        Density matrix of composite object
    selB : int/list
        Selected components for density matrix B
    base : {e, 2}, default: e
        Base of logarithm.
    sparse : bool, default: False
        Use sparse eigensolver.

    Returns
    -------
    ent_cond : float
        Value of conditional entropy

    """
    if rho.type != 'oper':
        raise TypeError("Input must be density matrix.")
    if isinstance(selB, int):
        selB = [selB]
    B = ptrace(rho, selB)
    out = (entropy_vn(rho, base, sparse=sparse) -
           entropy_vn(B, base, sparse=sparse))
    return out


def participation_ratio(rho):
    """
    Returns the effective number of states for a density matrix.

    The participation is unity for pure states, and maximally N,
    where N is the Hilbert space dimensionality, for completely
    mixed states.

    Parameters
    ----------
    rho : qobj
        Density matrix

    Returns
    -------
    pr : float
        Effective number of states in the density matrix

    """
    if rho.type == 'ket' or rho.type == 'bra':
        return 1.0
    else:
        return 1.0 / (rho ** 2).tr()


def entangling_power(U):
    """
    Calculate the entangling power of a two-qubit gate U, which
    is zero of nonentangling gates and 2/9 for maximally entangling gates.

    Parameters
    ----------
    U : qobj
        Qobj instance representing a two-qubit gate.

    Returns
    -------
    ep : float
        The entanglement power of U (real number between 0 and 2/9)

    References:

        Explorations in Quantum Computing, Colin P. Williams (Springer, 2011)
    """

    if not U.isoper:
        raise Exception("U must be an operator.")

    if U.dims != [[2, 2], [2, 2]]:
        raise Exception("U must be a two-qubit gate.")

    from qutip.core.gates import swap
    swap13 =  expand_operator(swap(dtype=U.dtype), [2, 2, 2, 2], [1, 3])
    a = tensor(U, U).dag() * swap13 * tensor(U, U) * swap13
    Uswap = swap(dtype=U.dtype) * U
    b = tensor(Uswap, Uswap).dag() * swap13 * tensor(Uswap, Uswap) * swap13

    return 5.0/9 - 1.0/36 * (a.tr() + b.tr()).real
