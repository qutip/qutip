import scipy.linalg
import scipy.sparse
from .dense import Dense
from .csr import CSR

__all__ = ['inv', 'inv_csr', 'inv_dense']


def inv_dense(data, /):
    """Compute the inverse of a matrix"""
    if not isinstance(data, Dense):
        raise TypeError("expected data in Dense format but got "
                        + str(type(data)))
    if data.shape[0] != data.shape[1]:
        raise ValueError('Cannot compute the matrix inverse'
                         ' of a nonsquare matrix')
    return Dense(scipy.linalg.inv(data.as_ndarray()), copy=False)


def inv_csr(data, /):
    """Compute the inverse of a sparse matrix"""
    if not isinstance(data, CSR):
        raise TypeError("expected data in CSR format but got "
                        + str(type(data)))
    if data.shape[0] != data.shape[1]:
        raise ValueError('Cannot compute the matrix inverse '
                         'of a nonsquare matrix')
    inv = scipy.sparse.linalg.inv(data.as_scipy().tocsc())
    # scipy.sparse.linalg.inv can return dense or sparse arrays.
    return CSR(scipy.sparse.csr_matrix(inv), copy=False)


from .dispatch import Dispatcher as _Dispatcher

inv = _Dispatcher(inv_dense, name='inv', inputs=('data',), out=True)
inv.__doc__ =\
    """
    Return matrix inverse for a data-layer object.

    Parameters
    ----------
    data : Data
        Input matrix

    Returns
    -------
    inverse : Data
        Inverse of data
    """
inv.add_specialisations([
    (CSR, CSR, inv_csr),
    (Dense, Dense, inv_dense),
], _defer=True)

del _Dispatcher
