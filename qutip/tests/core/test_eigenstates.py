import pytest
import numpy as np
import qutip
from itertools import combinations


def _canonicalise_eigenvector(vec):
    """
    Normalise an eigenvector so that the first non-zero value is equal to one,
    and the array is flattened.  Just normalising based on vector magnitude
    isn't enough to fully fix the gauge because the vectors could still be
    multiplied by a unit complex number.
    """
    vec = vec.flatten()
    nonzero = vec != 0
    if not np.any(nonzero):
        return vec
    return vec / vec[np.argmax(nonzero)]


# Random diagonal Hamiltonian.
_diagonal_dimension = 10
_diagonal_eigenvalues = np.sort(np.random.rand(_diagonal_dimension))
_diagonal_eigenstates = np.array([[0]*n + [1] + [0]*(_diagonal_dimension-n-1)
                                  for n in range(_diagonal_dimension)])
_diagonal_hamiltonian = qutip.qdiags(_diagonal_eigenvalues, 0)

# Arbitrary known non-diagonal complex Hamiltonian.
_nondiagonal_hamiltonian = qutip.Qobj(np.array([
    [0.16252356,             0.27696416+0.0405202j,  0.19577420+0.07815636j],
    [0.27696416-0.0405202j,  0.45859633,             0.36222915+0.17372725j],
    [0.19577420-0.07815636j, 0.36222915-0.17372725j, 0.44149665]]))
_nondiagonal_eigenvalues = np.array([
    -0.022062710138316392, 0.08888141616526818, 0.995797833973048])
_nondiagonal_eigenstates = np.array([
    [-0.737511505546763, 0.5270680510449308-0.29398599661318j,
     0.009793118179759598+0.3029065489313791j],
    [0.5552814080417957, 0.23570050756381764 - 0.3577691669342573j,
     -0.3741560255426259+0.6067259021655438j],
    [-0.3843687514214284, -0.670810624386174+0.04723455831286158j,
     -0.5593181579625106+0.2953063897306936j]])


@pytest.mark.parametrize(["hamiltonian", "eigenvalues", "eigenstates"], [
    pytest.param(qutip.sigmaz(), [-1, 1], [[0, 1], [1, 0]], id="diagonal-2"),
    pytest.param(_diagonal_hamiltonian, _diagonal_eigenvalues,
                 _diagonal_eigenstates,
                 id="diagonal-"+str(_diagonal_dimension)),
    pytest.param(qutip.sigmax(), [-1, 1], [[-1, 1], [1, 1]], id="sigmax"),
    pytest.param(_nondiagonal_hamiltonian, _nondiagonal_eigenvalues,
                 _nondiagonal_eigenstates, id="non-diagonal"),
])
def test_known_eigensystem(hamiltonian, eigenvalues, eigenstates):
    test_values, test_states = hamiltonian.eigenstates()
    eigenvalues = np.array(eigenvalues)
    eigenstates = np.array(eigenstates)
    test_order = np.argsort(test_values)
    test_vectors = [_canonicalise_eigenvector(test_states[i].full())
                    for i in test_order]
    expected_order = np.argsort(eigenvalues)
    expected_vectors = [_canonicalise_eigenvector(eigenstates[i])
                        for i in expected_order]
    np.testing.assert_allclose(test_values[test_order],
                               eigenvalues[expected_order],
                               atol=1e-10)
    for test, expected in zip(test_vectors, expected_vectors):
        np.testing.assert_allclose(test, expected, atol=1e-10)


# Specify parametrisation over a random Hamiltonian by specifying the
# dimensions, rather than duplicating that logic.
@pytest.fixture(params=[
    pytest.param([10], id="simple"),
    pytest.param([5, 3, 4], id="tensor"),
    pytest.param([3, 3, 3], id="degenerate")])
def random_hamiltonian(request):
    dimensions = request.param
    eigen = None
    dist = "fill"
    if dimensions == [3, 3, 3]:
        eigen = [0, 1, 1] * 9
        dist = "eigen"
    return qutip.rand_herm(dimensions, distribution=dist, eigenvalues=eigen)


@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('dtype', ['csr', 'dense'])
def test_satisfy_eigenvalue_equation(random_hamiltonian, sparse, dtype):
    random_hamiltonian = random_hamiltonian.to(dtype)
    eigenvalues, eigenstates = random_hamiltonian.eigenstates(sparse=sparse)
    for eigenvalue, eigenstate in zip(eigenvalues, eigenstates):
        np.testing.assert_allclose((random_hamiltonian * eigenstate).full(),
                                   (eigenvalue * eigenstate).full(),
                                   atol=1e-10)
        assert eigenstate.norm() == pytest.approx(1., abs=1e-10)
    errors = [evec1.overlap(evec2)
              for evec1, evec2 in combinations(eigenstates, 2)]
    assert np.max(np.abs(errors)) == pytest.approx(0., abs=1e-8)
