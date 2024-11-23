import numpy as np
from qutip import basis, tensor
import pytest
from qutip.distributions import (
    WignerDistribution, QDistribution,
    TwoModeQuadratureCorrelation, HarmonicOscillatorWaveFunction,
    HarmonicOscillatorProbabilityFunction
)

@pytest.fixture
def harmonic_oscillator_ground_state():
    return np.array([[1.0 + 0.0j]])  

@pytest.fixture
def harmonic_oscillator_first_excited_state():
    return np.array([[0.0 + 0.0j, 1.0 + 0.0j]])

@pytest.fixture
def two_mode_wavefunction():
    psi1 = basis(5, 0)
    psi2 = basis(5, 0)
    return tensor(psi1, psi2)

@pytest.fixture
def two_mode_density_matrix(two_mode_wavefunction):
    return two_mode_wavefunction.proj()

@pytest.fixture
def density_matrix():
    psi = basis(5, 0)
    return psi.proj()

def test_wigner_distribution_update(density_matrix):
    dist = WignerDistribution()
    dist.update(density_matrix)
    assert dist.data is not None
    assert dist.data.shape == (250, 250)
    assert np.isclose(dist.data.sum() * (dist.xvecs[0][1] - dist.xvecs[0][0]) ** 2, 1.0, atol=1e-2)


def test_q_distribution_update(density_matrix):
    dist = QDistribution()
    dist.update(density_matrix)
    assert dist.data is not None
    assert dist.data.shape == (250, 250)
    assert dist.data.min() >= 0 
    assert np.isclose(dist.data.sum() * (dist.xvecs[0][1] - dist.xvecs[0][0]) ** 2, 1.0, atol=1e-2)

def test_harmonic_oscillator_initialization():
    wavefunc = HarmonicOscillatorWaveFunction()
    assert wavefunc.xlabels ==[r'$x$']
    assert wavefunc.omega == 1.0
    assert len(wavefunc.xvecs) == 1
    assert wavefunc.xvecs[0].shape == (250,)

def test_harmonic_oscillator_wavefunction_ground_state(harmonic_oscillator_ground_state):
    
    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_ground_state)

    print(wavefunc.data)
    print(wavefunc.data[::-1])

    assert wavefunc.data is not None
    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))  

    assert np.allclose(wavefunc.data.imag, 0.0, atol=1e-10)
    assert np.allclose(wavefunc.data, wavefunc.data[::-1], atol=1e-10)

def test_harmonic_oscillator_wavefunction_first_excited_state(harmonic_oscillator_first_excited_state):

    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_first_excited_state)

    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))

    assert np.allclose(wavefunc.data.imag, 0.0, atol=1e-10) 

    midpoint = len(wavefunc.data) // 2
    if len(wavefunc.data) % 2 == 0:
        left = wavefunc.data[:midpoint]
        right = wavefunc.data[midpoint:][::-1]
    else:
        left = wavefunc.data[:midpoint]
        right = wavefunc.data[midpoint + 1:][::-1]

    assert np.allclose(left, -right, atol=1e-10)

def test_harmonic_oscillator_with_custom_extent(harmonic_oscillator_ground_state):

    extent = [-10, 10]
    steps = 500
    wavefunc = HarmonicOscillatorWaveFunction(extent=extent, steps=steps)
    wavefunc.update(harmonic_oscillator_ground_state)

    assert wavefunc.xvecs[0][0] == extent[0]
    assert wavefunc.xvecs[0][-1] == extent[1]
    assert wavefunc.xvecs[0].shape == (steps,)

def test_harmonic_oscillator_high_excited_state():

    high_state = np.zeros((1, 50), dtype=complex)
    high_state[0, -1] = 1.0 + 0.0j

    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(high_state)

    assert wavefunc.data.shape == (250,)
    assert np.all(np.isfinite(wavefunc.data))

def test_harmonic_oscillator_wavefunction_normalization(harmonic_oscillator_ground_state):

    wavefunc = HarmonicOscillatorWaveFunction()
    wavefunc.update(harmonic_oscillator_ground_state)

    dx = wavefunc.xvecs[0][1] - wavefunc.xvecs[0][0]
    integral = np.sum(np.abs(wavefunc.data) ** 2) * dx

    assert np.isclose(integral, 1.0, atol=1e-2)

def test_harmonic_oscillator_probability_function_update(density_matrix):
    prob_func = HarmonicOscillatorProbabilityFunction()
    prob_func.update(density_matrix)
    assert len(prob_func.xvecs) == 1
    assert prob_func.data.shape == (250,)  
    assert prob_func.data.min() >= 0  
    assert np.isclose(prob_func.data.sum()*(prob_func.xvecs[0][1] - prob_func.xvecs[0][0]), 1.0, atol=1e-2) 

def test_two_mode_quadrature_correlation_update(two_mode_wavefunction):

    corr = TwoModeQuadratureCorrelation()
    corr.update(two_mode_wavefunction)

    assert corr.data.shape == (250, 250)

    assert corr.data.min() >= 0

    dx1 = corr.xvecs[0][1] - corr.xvecs[0][0]
    dx2 = corr.xvecs[1][1] - corr.xvecs[1][0]
    integral = corr.data.sum() * dx1 * dx2
    assert np.isclose(integral, 1.0, atol=1e-2)
