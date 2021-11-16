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
    HEOMSolver,
    BosonicHEOMSolver,
    FermionicHEOMSolver,
    HSolverDL,
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


class TestSolverAPI:
    def test_api(self):
        # just assert that the baths are importable
        assert HEOMSolver
        assert BosonicHEOMSolver
        assert FermionicHEOMSolver
        assert HSolverDL
