import scipy.sparse as sp

from .. import Qobj

__all__ = ["sparray_to_spmatrix", "qobj_from_sparray"]


def sparray_to_spmatrix(sparr, format="csr"):
    """Convert a SciPy sparse *array* to an equivalent sparse *matrix*.

    Rebuilt from the COO triplet, so it works for any sparse-array format and
    never triggers the ``block_diag``-style "switching to the sparse array
    interface" warning. Crucially this keeps the data **sparse** (unlike
    ``toarray``), so a ``Qobj`` built from the result uses QuTiP's sparse CSR
    layer rather than the dense one.
    """
    coo = sparr.tocoo()
    coords = getattr(coo, "coords", None)
    if coords is None:                       # older sparse-array builds
        coords = (coo.row, coo.col)
    return sp.coo_matrix((coo.data, coords), shape=coo.shape).asformat(format)


def qobj_from_sparray(sparr):
    """Build a sparse-backed :class:`.Qobj` from a SciPy sparse *array*."""
    return Qobj(sparray_to_spmatrix(sparr))
