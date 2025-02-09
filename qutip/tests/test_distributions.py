import numpy as np
from qutip import basis, tensor
import pytest
from qutip.distributions import (
    TwoModeQuadratureCorrelation, HarmonicOscillatorWaveFunction,
    HarmonicOscillatorProbabilityFunction
)

# Fixtures provide reusable test data for the test functions below.

@pytest.fixture
def harmonic_oscillator_ground_state():
    """
    Fixture for the ground state wavefunction of a harmonic oscillator.
    
    Returns:
        np.ndarray: Ground state wavefunction as a 2D complex array.
    """
    return basis(1,0)

@pytest.fixture
def harmonic_oscillator_first_excited_state():
    """
    Fixture for the first excited state wavefunction of a harmonic oscillator.
    
    Returns:
        np.ndarray: First excited state wavefunction as a 2D complex array.
    """
    return basis(2,1)

@pytest.fixture
def two_mode_wavefunction():
    """
    Fixture for the wavefunction of a two-mode quantum system.
    
    Returns:
        Qobj: Tensor product of the ground states for two modes.
    """
    psi1 = basis(5, 0)  # Ground state for mode 1
    psi2 = basis(5, 0)  # Ground state for mode 2
    return tensor(psi1, psi2)

@pytest.fixture
def two_mode_density_matrix(two_mode_wavefunction):
    """
    Fixture for the density matrix of a two-mode quantum system.
    
    Args:
        two_mode_wavefunction (Qobj): Wavefunction of the two-mode system.

    Returns:
        Qobj: Density matrix for the two-mode system.
    """
    return two_mode_wavefunction.proj()

@pytest.fixture
def density_matrix():
    """
    Fixture for the density matrix of a single-mode quantum system.
    
    Returns:
        Qobj: Density matrix for the ground state.
    """
    psi = basis(5, 0)  # Ground state
    return psi.proj()

# Tests for initializing the harmonic oscillator wavefunction.

def test_harmonic_oscillator_initialization():
    """
    Tests the initialization of the Harmonic Oscillator WaveFunction object.
    """
    wavefunc = HarmonicOscillatorWaveFunction()
    assert wavefunc.xlabels == [r'$x$']
    assert wavefunc.omega == 1.0
    assert len(wavefunc.xvecs) == 1
    assert wavefunc.xvecs[0].shape == (250,)

# Tests for ground state properties of the harmonic oscillator wavefunction.

def test_harmonic_oscillator_wavefunction_ground_state(harmonic_oscillator_ground_state):
    """
    Validates properties of the harmonic oscillator's ground state wavefunction.

    Args:
        harmonic_oscillator_ground_state (np.ndarray): Ground state wavefunction.
    """
    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_ground_state)

    assert wavefunc.data is not None
    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))  
    assert np.allclose(wavefunc.data.imag, 0.0, atol=1e-10)
    assert np.allclose(wavefunc.data, wavefunc.data[::-1], atol=1e-10)

# Tests for the first excited state properties.

def test_harmonic_oscillator_wavefunction_first_excited_state(harmonic_oscillator_first_excited_state):
    """
    Validates properties of the harmonic oscillator's first excited state.

    Args:
        harmonic_oscillator_first_excited_state (np.ndarray): First excited state wavefunction.
    """
    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_first_excited_state)

    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))
    assert np.allclose(wavefunc.data.imag, 0.0, atol=1e-10) 

    # Symmetry check: odd function property of the first excited state
    midpoint = len(wavefunc.data) // 2
    if len(wavefunc.data) % 2 == 0:
        left = wavefunc.data[:midpoint]
        right = wavefunc.data[midpoint:][::-1]
    else:
        left = wavefunc.data[:midpoint]
        right = wavefunc.data[midpoint + 1:][::-1]

    assert np.allclose(left, -right, atol=1e-10)

# Tests with custom spatial extents.

def test_harmonic_oscillator_with_custom_extent(harmonic_oscillator_ground_state):
    """
    Tests wavefunction initialization with custom spatial extent and steps.

    Args:
        harmonic_oscillator_ground_state (np.ndarray): Ground state wavefunction.
    """
    extent = [-10, 10]
    steps = 500
    wavefunc = HarmonicOscillatorWaveFunction(extent=extent, steps=steps)
    wavefunc.update(harmonic_oscillator_ground_state)

    assert wavefunc.xvecs[0][0] == extent[0]
    assert wavefunc.xvecs[0][-1] == extent[1]
    assert wavefunc.xvecs[0].shape == (steps,)

# Test the wavefunction for a highly excited state of the harmonic oscillator.

def test_harmonic_oscillator_high_excited_state():
    """
    Tests the wavefunction for a highly excited harmonic oscillator state.

    Ensures the wavefunction has the correct shape and contains valid finite values.
    """
    high_state = np.zeros((1, 50), dtype=complex)
    high_state[0, -1] = 1.0 + 0.0j

    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(high_state)

    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))


# Tests for normalization of wavefunctions.

def test_harmonic_oscillator_wavefunction_normalization(harmonic_oscillator_ground_state):
    """
    Tests the normalization of a harmonic oscillator wavefunction.

    Args:
        harmonic_oscillator_ground_state (np.ndarray): Ground state wavefunction.
    """
    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_ground_state)

    dx = wavefunc.xvecs[0][1] - wavefunc.xvecs[0][0]
    integral = np.sum(np.abs(wavefunc.data) ** 2) * dx

    assert np.isclose(integral, 1.0, atol=1e-2)

# Tests for probability functions.

def test_harmonic_oscillator_probability_function_update(density_matrix):
    """
    Validates the update of a harmonic oscillator's probability function.

    Args:
        density_matrix (Qobj): Density matrix of the quantum system.
    """
    prob_func = HarmonicOscillatorProbabilityFunction()
    prob_func.update(density_matrix)
    assert len(prob_func.xvecs) == 1
    assert prob_func.data.shape == (250,)  
    assert prob_func.data.min() >= 0  
    assert np.isclose(prob_func.data.sum() * (prob_func.xvecs[0][1] - prob_func.xvecs[0][0]), 1.0, atol=1e-2) 

# Tests for two-mode quantum system properties.

def test_two_mode_quadrature_correlation_update(two_mode_wavefunction):
    """
    Validates the update of the two-mode quadrature correlation.

    Args:
        two_mode_wavefunction (Qobj): Wavefunction of the two-mode system.
    """
    corr = TwoModeQuadratureCorrelation()
    corr.update(two_mode_wavefunction)

    assert corr.data.shape == (250, 250)
    assert corr.data.min() >= 0

    dx1 = corr.xvecs[0][1] - corr.xvecs[0][0]
    dx2 = corr.xvecs[1][1] - corr.xvecs[1][0]
    integral = corr.data.sum() * dx1 * dx2
    assert np.isclose(integral, 1.0, atol=1e-2)
