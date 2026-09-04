import pytest

from qutip.testing import random_data
from qutip.core.data import Dense, CSR, Dia
from qutip.core import data

# Set up the special cases for each type of matrix that will be tested.  These
# should be kept low, because mathematical operations will test a Cartesian
# product of all the cases of the same order as the operation, which can get
# very large very fast.  The operations should each complete in a small amount
# of time, so having 10000+ tests in this file still ought to take less than 2
# minutes, but it's easy to accidentally add orders of magnitude on.


def cases_csr(shape):
    """
    Return a list of generators of the different special cases for CSR
    matrices of a given shape.
    """
    def factory(density, sort):
        return lambda gen: random_data.random_csr(shape, density, sort, gen)

    def zero_factory():
        return lambda _: data.csr.zeros(shape[0], shape[1])
    return [
        pytest.param(factory(0.001, True), id="sparse"),
        pytest.param(factory(0.8, True), id="filled,sorted"),
        pytest.param(factory(0.8, False), id="filled,unsorted"),
        pytest.param(zero_factory(), id="zero"),
    ]


def cases_dense(shape):
    """
    Return a list of generators of the different special cases for Dense
    matrices of a given shape.
    """
    def factory(fortran):
        return lambda gen: random_data.random_dense(shape, fortran, gen)
    return [
        pytest.param(factory(False), id="C"),
        pytest.param(factory(True), id="Fortran"),
    ]


def cases_diag(shape):
    """
    Return a list of generators of the different special cases for Dense
    matrices of a given shape.
    """
    def factory(density, sort=False):
        return lambda gen: random_data.random_diag(shape, density, sort, gen)

    def zero_factory():
        return lambda _: data.dia.zeros(shape[0], shape[1])

    return [
        pytest.param(factory(0.001), id="sparse"),
        pytest.param(factory(0.8, True), id="filled,sorted"),
        pytest.param(factory(0.8, False), id="filled,unsorted"),
        pytest.param(zero_factory(), id="zero"),
    ]


CORRECT_CASES = {
    CSR: cases_csr,
    Dia: cases_diag,
    Dense: cases_dense,
}

WRONG_CASES = {
    CSR: lambda shape: [lambda gen: random_data.random_csr(shape, 0.5, True, gen)],
    Dense: lambda shape: [lambda gen: random_data.random_dense(shape, False, gen)],
    Dia: lambda shape: [lambda gen: random_data.random_diag(shape, 0.5, gen=gen)],
}
