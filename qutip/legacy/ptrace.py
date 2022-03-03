__all__ = []

import numpy as np
import scipy.sparse as sp
from qutip.sparse import sp_reshape


def _ptrace(rho, sel):
    """
    Private function calculating the partial trace.
    """

    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho.dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    drho = rho.dims[0]
    N = np.prod(drho)
    M = np.prod(np.asarray(drho).take(sel))

    if np.prod(rho.dims[1]) == 1:
        rho = rho * rho.dag()

    perm = sp.lil_matrix((M * M, N * N))
    # all elements in range(len(drho)) not in sel set
    rest = np.setdiff1d(np.arange(len(drho)), sel)
    ilistsel = _select(sel, drho)
    indsel = _list2ind(ilistsel, drho)
    ilistrest = _select(rest, drho)
    indrest = _list2ind(ilistrest, drho)
    irest = (indrest - 1) * N + indrest - 2
    # Possibly use parfor here if M > some value ?
    # We have to initialise like this to get a numpy array of lists rather than
    # a standard numpy 2d array.  Scipy >= 1.5 requires that we do this, rather
    # than just pass a 2d array, and scipy < 1.5 will accept it (because it's
    # actually the correct format).
    perm.rows = np.empty((M * M,), dtype=object)
    perm.data = np.empty((M * M,), dtype=object)
    for m in range(M * M):
        perm.rows[m] = list((irest
                             + (indsel[int(np.floor(m / M))] - 1)*N
                             + indsel[int(np.mod(m, M))]).T[0])
        perm.data[m] = [1.0] * len(perm.rows[m])
    perm = perm.tocsr()
    rhdata = perm * sp_reshape(rho.data, (np.prod(rho.shape), 1))
    rho1_data = sp_reshape(rhdata, (M, M))
    dims_kept0 = np.asarray(rho.dims[0]).take(sel)
    dims_kept1 = np.asarray(rho.dims[0]).take(sel)
    rho1_dims = [dims_kept0.tolist(), dims_kept1.tolist()]
    rho1_shape = [np.prod(dims_kept0), np.prod(dims_kept1)]
    return rho1_data, rho1_dims, rho1_shape


def _list2ind(ilist, dims):
    """!
    Private function returning indicies
    """
    ilist = np.asarray(ilist)
    dims = np.asarray(dims)
    irev = np.fliplr(ilist) - 1
    fact = np.append(np.array([1]), (np.cumprod(np.flipud(dims)[:-1])))
    fact = fact.reshape(len(fact), 1)
    return np.array(np.sort(np.dot(irev, fact) + 1, 0), dtype=int)


def _select(sel, dims):
    """
    Private function finding selected components
    """
    sel = np.asarray(sel)  # make sure sel is np.array
    dims = np.asarray(dims)  # make sure dims is np.array
    rlst = dims.take(sel)
    rprod = np.prod(rlst)
    ilist = np.ones((rprod, len(dims)), dtype=int)
    counter = np.arange(rprod)
    for k in range(len(sel)):
        ilist[:, sel[k]] = np.remainder(
            np.fix(counter / np.prod(dims[sel[k + 1:]])), dims[sel[k]]) + 1
    return ilist
