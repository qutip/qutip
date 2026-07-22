"""
Helper module for the use of ``scipy.sparse`` backing containers in the
data layer.

In the light of SciPy's migration plans, ``spmatrix`` types (``csr_matrix``,
``dia_matrix``, ...) will be replaced by ``sparray`` types.
We respond to this by ensuring that qutip's data layer always uses ``sparray``
containers internally.
A legacy matrix is only produced when a user explicitly requests one
via such public methods as ``extract`` / ``Qobj.data_as``.

In our migration strategy, we want to ensure (hence this helper module) that:

* We detect SciPy's CSR/Dia objects regardless of the exact
  implementation (csr_matrix/csr_array or dia_matrix/dia_array).
* We correctly convert array to matrix views (upon user request). This
  is to ensure that before SciPy officially deprecates sparse matrix,
  users of qutip can still rely on its interface in their downstream
  code.

We also introduce an additional guard that protects qutip's imports from breaking
for the future scenario when SciPy will drop legacy matrix types
(see ``_legacy_matrix_type``).
"""

from scipy.sparse import issparse

try:
    from scipy.sparse import csr_matrix, dia_matrix

    _LEGACY_MATRIX_TYPES = {"csr": csr_matrix, "dia": dia_matrix}
except ImportError:
    _LEGACY_MATRIX_TYPES = {}

__all__ = [
    "is_csr",
    "is_dia",
    "csr_as_matrix",
    "dia_as_matrix",
]

_SCIPY_SPARSE_ROADMAP = (
    "https://docs.scipy.org/doc/scipy/dev/roadmap-detailed.html#sparse"
)


def _legacy_matrix_type(kind):
    """
    Return the legacy ``{kind}_matrix`` class, or raise a clear error if the
    installed SciPy no longer ships it.

    Ensures qutip's forward compatibility with
    SciPy's removal of the ``spmatrix`` types: as long as SciPy provides the
    class we hand it back, and once SciPy drops it a user who explicitly asked
    for a matrix gets an actionable exception message.
    """
    try:
        return _LEGACY_MATRIX_TYPES[kind]
    except KeyError:
        raise TypeError(
            f"This SciPy build no longer provides "
            f"'scipy.sparse.{kind}_matrix'; request the '{kind}_array' output "
            f"instead. See {_SCIPY_SPARSE_ROADMAP}."
        ) from None


def is_csr(arg):
    """``True`` for a scipy CSR matrix or array (format-agnostic)."""
    return issparse(arg) and arg.format == "csr"


def is_dia(arg):
    """``True`` for a scipy DIA matrix or array (format-agnostic)."""
    return issparse(arg) and arg.format == "dia"


def csr_as_matrix(csr):
    """
    Return a legacy ``csr_matrix`` viewing the same memory buffers as
    ``csr``.

    Internal storage containers in the data layer are always
    ``csr_array``. This function rebuilds a ``csr_matrix`` over the same
    ``data``/``indices``/``indptr`` (no copy) in case of a user
    explicitly requests a matrix.
    """
    return _legacy_matrix_type("csr")(
        (csr.data, csr.indices, csr.indptr), shape=csr.shape, copy=False
    )


def dia_as_matrix(dia):
    """
    Return a legacy ``dia_matrix`` viewing the same memory buffers as
    ``dia``.

    The ``dia_array`` counterpart of :func:`csr_as_matrix`.
    """
    return _legacy_matrix_type("dia")(
        (dia.data, dia.offsets), shape=dia.shape, copy=False
    )
