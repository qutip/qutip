"""
Tests for qutip.nonmarkov.heom.
"""

from qutip.nonmarkov.heom import (
    BathExponent,
    Bath,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
    UnderDampedBath,
    FermionicBath,
    LorentzianBath,
    LorentzianPadeBath,
    HEOMSolver,
    HSolverDL,
    HierarchyADOs,
    HierarchyADOsState,
)


class TestBathAPI:
    def test_api(self):
        # just assert that the baths are importable
        assert BathExponent
        assert Bath
        assert BosonicBath
        assert DrudeLorentzBath
        assert DrudeLorentzPadeBath
        assert UnderDampedBath
        assert FermionicBath
        assert LorentzianBath
        assert LorentzianPadeBath


class TestSolverAPI:
    def test_api(self):
        # just assert that the solvers and associated classes are importable
        assert HEOMSolver
        assert HSolverDL
        assert HierarchyADOs
        assert HierarchyADOsState
