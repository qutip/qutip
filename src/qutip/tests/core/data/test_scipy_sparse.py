"""
Tests for qutip.core.data._scipy_sparse, in particular the forward-compatible
"removal path": the behaviour once a future SciPy stops shipping the legacy
``csr_matrix`` / ``dia_matrix`` types.

qutip stores everything internally as the modern ``*_array`` types, so the only
place that needs the legacy matrices is the explicit user request for one
(``Qobj.data_as("csr_matrix")`` / ``data.extract(..., "csr_matrix")``).  We
cannot actually uninstall the types from SciPy, so we simulate their removal by
emptying the ``_LEGACY_MATRIX_TYPES`` registry that :mod:`_scipy_sparse`
resolves once at import time.
"""
import importlib

import pytest

from qutip.core import data
from qutip.core.data import _scipy_sparse
from qutip import qeye


@pytest.fixture
def scipy_without_legacy_matrix(monkeypatch):
    """Simulate a SciPy build that no longer provides the ``spmatrix`` types."""
    monkeypatch.setattr(_scipy_sparse, "_LEGACY_MATRIX_TYPES", {})


@pytest.mark.parametrize("kind", ["csr", "dia"])
def test_legacy_matrix_type_available(kind):
    """While SciPy ships the legacy types, they are handed back unchanged."""
    import scipy.sparse

    cls = _scipy_sparse._legacy_matrix_type(kind)
    assert cls is getattr(scipy.sparse, f"{kind}_matrix")


@pytest.mark.parametrize("kind", ["csr", "dia"])
def test_legacy_matrix_type_removed_raises(kind, scipy_without_legacy_matrix):
    """Once SciPy drops the type, the request raises a clear, actionable error."""
    with pytest.raises(TypeError) as exc_info:
        _scipy_sparse._legacy_matrix_type(kind)
    message = str(exc_info.value)
    # The message must name the missing type and point at the array replacement
    # and the migration roadmap, so a downstream user knows how to react.
    assert f"{kind}_matrix" in message
    assert f"{kind}_array" in message
    assert _scipy_sparse._SCIPY_SPARSE_ROADMAP in message


def test_csr_as_matrix_removed_raises(scipy_without_legacy_matrix):
    scipy_csr = qeye(2, dtype="csr")._data.as_scipy()
    with pytest.raises(TypeError, match="csr_matrix"):
        _scipy_sparse.csr_as_matrix(scipy_csr)


def test_dia_as_matrix_removed_raises(scipy_without_legacy_matrix):
    scipy_dia = qeye(2, dtype="dia")._data.as_scipy()
    with pytest.raises(TypeError, match="dia_matrix"):
        _scipy_sparse.dia_as_matrix(scipy_dia)


@pytest.mark.parametrize("dtype, fmt", [
    pytest.param("csr", "csr_matrix", id="csr"),
    pytest.param("dia", "dia_matrix", id="dia"),
])
def test_extract_legacy_matrix_removed_raises(
    dtype, fmt, scipy_without_legacy_matrix
):
    """End-to-end: an explicit legacy-matrix extraction surfaces the error."""
    matrix = qeye(2, dtype=dtype)
    with pytest.raises(TypeError, match=fmt):
        data.extract(matrix._data, fmt)


@pytest.mark.parametrize("dtype, fmt", [
    pytest.param("csr", "csr_array", id="csr"),
    pytest.param("dia", "dia_array", id="dia"),
])
def test_extract_array_still_works_after_removal(
    dtype, fmt, scipy_without_legacy_matrix
):
    """The modern ``*_array`` path is unaffected by the legacy types going away."""
    matrix = qeye(2, dtype=dtype)
    result = data.extract(matrix._data, fmt)
    assert result.format == dtype


@pytest.fixture
def scipy_module_without_legacy_matrix(monkeypatch):
    """
    Simulate importing qutip against a SciPy that has *removed* the legacy
    matrix types, exercising the module-body ``except ImportError`` branch.

    We delete the classes from ``scipy.sparse`` so that the ``from
    scipy.sparse import csr_matrix, dia_matrix`` at the top of
    :mod:`_scipy_sparse` fails, then re-execute the module.  The module object
    is reused across the reload, so functions imported elsewhere keep seeing a
    consistent (restored) module namespace afterwards.
    """
    import scipy.sparse

    monkeypatch.delattr(scipy.sparse, "csr_matrix", raising=False)
    monkeypatch.delattr(scipy.sparse, "dia_matrix", raising=False)
    try:
        importlib.reload(_scipy_sparse)
        yield
    finally:
        # monkeypatch restores the deleted attributes at teardown; reload once
        # more so the live module reflects a SciPy that ships the types again.
        monkeypatch.undo()
        importlib.reload(_scipy_sparse)


def test_import_never_breaks_when_scipy_removes_legacy(
    scipy_module_without_legacy_matrix,
):
    """
    Importing the module must succeed even when SciPy no longer provides the
    legacy types: the registry is simply empty and the public API is intact.
    """
    assert _scipy_sparse._LEGACY_MATRIX_TYPES == {}
    for name in _scipy_sparse.__all__:
        assert hasattr(_scipy_sparse, name)
    # The legacy request path is still importable and only fails when actually
    # called -- it must not raise at import time.
    with pytest.raises(TypeError, match="csr_matrix"):
        _scipy_sparse._legacy_matrix_type("csr")


def test_qutip_operations_work_when_scipy_removes_legacy(
    scipy_module_without_legacy_matrix,
):
    """Normal qutip usage (which only needs ``*_array``) is unaffected."""
    result = qeye(2, dtype="csr") @ qeye(2, dtype="csr")
    assert result.data_as("csr_array").format == "csr"
