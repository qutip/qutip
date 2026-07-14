"""
Helper for qutip's use of ``scipy.sparse`` containers.

qutip uses scipy sparse purely as a storage/interoperability container.

In the light of SciPy's migration plans, ``spmatrix`` types will be replaced by the new (SciPy >= 1.8) ``sparray`` types.
We adopt to this migration strategy by ensuring that qutip's data layer always builds ``sparray`` containers, and a legacy matrix is only produced when a user explicitly requests one at such public access points as ``extract`` / ``Qobj.data_as``.

In our migration strategy, we want to ensure (hence this module) that:

* we do format-agnostic type detection. We aim to recognise a CSR/DIA object regardless of whether it is a legacy matrix or a new array. 
* we correctly convert array to matrix views (upon user request). This is to ensure that before SciPy officially deprecates sparse matrix, users of qutip can still rely on its interface in their downstream code.
"""

from scipy.sparse import issparse, csr_matrix, dia_matrix

__all__ = [
    'is_csr', 'is_dia',
    'csr_as_matrix', 'dia_as_matrix',
]


def is_csr(arg):
    """``True`` for a scipy CSR matrix or array (format-agnostic)."""
    return issparse(arg) and arg.format == "csr"


def is_dia(arg):
    """``True`` for a scipy DIA matrix or array (format-agnostic)."""
    return issparse(arg) and arg.format == "dia"


def csr_as_matrix(csr):
    """
    Return a legacy ``csr_matrix`` viewing the same memory buffers as ``csr``.

    Internal storage containers in the data layer are always ``csr_array``.
    This function rebuilds a ``csr_matrix`` over the same ``data``/``indices``/``indptr`` (no copy) in case of a user explicitly requests a matrix.
    """
    return csr_matrix(
        (csr.data, csr.indices, csr.indptr), shape=csr.shape, copy=False
    )


def dia_as_matrix(dia):
    """
    Return a legacy ``dia_matrix`` viewing the same memory buffers as ``dia``.

    The ``dia_array`` counterpart of :func:`csr_as_matrix`.
    """
    return dia_matrix((dia.data, dia.offsets), shape=dia.shape, copy=False)
