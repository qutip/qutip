"""
Shared fixtures and markers for HEOM tests.
"""

import pytest

petsc4py_installed = False
try:
    import petsc4py  # noqa: F401
    from petsc4py import PETSc  # noqa: F401
    petsc4py_installed = True
except ImportError:
    pass

requires_petsc4py = pytest.mark.skipif(
    not petsc4py_installed,
    reason="petsc4py is not installed",
)
