"""
This module provides solvers for system-bath evoluation using the
HEOM (hierarchy equations of motion).

See https://en.wikipedia.org/wiki/Hierarchical_equations_of_motion for a very
basic introduction to the technique.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.

For backwards compatibility with QuTiP 4.6 and below, a new version of
HSolverDL (the Drude-Lorentz specific HEOM solver) is provided. It is
implemented on top of the new HEOMSolver but should largely be a drop-in
replacement for the old HSolverDL.

An exact copy of the QuTiP 4.6 HSolverDL is provided in
``qutip.nonmarkov.dlheom_solver`` for cases where the functionality of the
older solver is required. The older solver will be completely removed in
QuTiP 5.
"""

__all__ = [
    "BathExponent",
    "Bath",
    "BosonicBath",
    "DrudeLorentzBath",
    "DrudeLorentzPadeBath",
    "UnderDampedBath",
    "FermionicBath",
    "LorentzianBath",
    "LorentzianPadeBath",
    "HEOMSolver",
    "HSolverDL",
    "HierarchyADOs",
    "HierarchyADOsState",
]

from .bofin_baths import (
    BathExponent,
    Bath,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
    UnderDampedBath,
    FermionicBath,
    LorentzianBath,
    LorentzianPadeBath,
)

from .bofin_solvers import (
    HEOMSolver,
    HSolverDL,
    HierarchyADOs,
    HierarchyADOsState,
)
