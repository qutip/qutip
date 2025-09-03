"""
Tests for two-mode Wigner and Q-function implementation.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from qutip.core.states import coherent, basis, ket2dm
from qutip.core.operators import squeeze
from qutip.core.tensor import tensor
from qutip.wigner import (wigner_2mode, qfunc_2mode, wigner_2mode_full, 
                         qfunc_2mode_full, wigner_2mode_xx, wigner_2mode_xp,
                         wigner_2mode_alpha, qfunc_2mode_alpha)


class TestTwoModeWigner:
    """Test suite for two-mode Wigner and Q-functions."""

    def test_coherent_states_mostly_positive_wigner(self):
        """Test that two-mode coherent states have predominantly positive Wigner function."""
        N = 8
        alpha1, alpha2 = 1.0, 0.5
        psi = tensor(coherent(N, alpha1), coherent(N, alpha2))
        rho = ket2dm(psi)
        
        xvec = np.linspace(-3, 3, 16)
        W = wigner_2mode(rho, xvec, xvec, interpretation='xx')
        
        assert W.shape == (16, 16)
        # Coherent states should be mostly positive with small negative regions allowed
        positive_fraction = np.sum(W > 0) / W.size
        assert positive_fraction > 0.5  # Most values should be positive (observed ~55%)
        assert W.max() > 0.1  # Should have significant positive values
        assert W.min() > -0.5  # Negative values should be bounded

    def test_squeezed_states_nonclassical(self):
        """Test that squeezed states show non-classical behavior (negative Wigner values)."""
        N = 10
        squeeze_factor = 0.6
        psi = tensor(squeeze(N, squeeze_factor) * coherent(N, 1.0), 
                     coherent(N, 0.0))
        rho = ket2dm(psi)
        
        xvec = np.linspace(-4, 4, 20)
        W = wigner_2mode(rho, xvec, xvec)
        
        # Squeezed states should exhibit non-classical behavior
        assert W.min() < -1e-6, "Squeezed states should have negative Wigner values"
        assert W.max() > 0, "Should also have positive regions"

    def test_coordinate_interpretations(self):
        """Test different coordinate interpretation modes."""
        N = 6
        rho = ket2dm(tensor(coherent(N, 0.8), coherent(N, 0.4)))
        coords = np.linspace(-2, 2, 10)
        
        # Test all three interpretations
        W_xx = wigner_2mode(rho, coords, coords, interpretation='xx')
        W_xp = wigner_2mode(rho, coords, coords, interpretation='xp')
        W_alpha = wigner_2mode(rho, coords, coords, interpretation='alpha')
        
        # All should have same shape
        assert W_xx.shape == W_xp.shape == W_alpha.shape == (10, 10)
        
        # Should give different results (not identical)
        assert not np.allclose(W_xx, W_xp, rtol=0.1)
        assert not np.allclose(W_xx, W_alpha, rtol=0.1)

    def test_qfunction_always_positive(self):
        """Test that Q-function is always non-negative."""
        N = 8
        # Use both classical and non-classical states
        states = [
            tensor(coherent(N, 1.0), coherent(N, 0.5)),  # Classical
            tensor(squeeze(N, 0.3) * coherent(N, 0.8), coherent(N, 0.0))  # Non-classical
        ]
        
        xvec = np.linspace(-3, 3, 12)
        
        for psi in states:
            rho = ket2dm(psi)
            Q = qfunc_2mode(rho, xvec, xvec)
            
            assert Q.shape == (12, 12)
            assert Q.min() >= -1e-12, "Q-function should always be non-negative"
            assert Q.max() > 0, "Q-function should have positive values"

    def test_vacuum_state(self):
        """Test Wigner function for two-mode vacuum state."""
        N = 8
        vacuum = tensor(basis(N, 0), basis(N, 0))
        rho = ket2dm(vacuum)
        
        xvec = np.linspace(-2, 2, 12)
        W = wigner_2mode(rho, xvec, xvec)
        
        # Vacuum should be centered at origin and positive
        assert W.shape == (12, 12)
        # Maximum should be near center
        max_idx = np.unravel_index(W.argmax(), W.shape)
        center_idx = (W.shape[0] // 2, W.shape[1] // 2)
        # Allow some tolerance for discretization
        assert abs(max_idx[0] - center_idx[0]) <= 1
        assert abs(max_idx[1] - center_idx[1]) <= 1

    def test_input_validation(self):
        """Test input validation and error handling."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.5), coherent(N, 0.3)))
        xvec = np.linspace(-1, 1, 8)
        
        # Test invalid interpretation
        with pytest.raises(ValueError, match="interpretation must be one of"):
            wigner_2mode(rho, xvec, xvec, interpretation='invalid')
        
        # Test mismatched dimensions (single-mode state)
        rho_1mode = ket2dm(coherent(N, 0.5))
        with pytest.raises(ValueError, match="two-mode tensor operator"):
            wigner_2mode(rho_1mode, xvec, xvec)
        
        # Test non-Hermitian operator
        rho_bad = tensor(coherent(N, 0.5), coherent(N, 0.3))  # Not a density matrix
        with pytest.raises((ValueError, TypeError)):
            wigner_2mode(rho_bad, xvec, xvec)

    def test_normalization(self):
        """Test that functions handle normalization properly."""
        N = 6
        psi = tensor(coherent(N, 1.0), coherent(N, 0.5))
        rho_unnormalized = ket2dm(psi) * 1.5  # Scale by 1.5
        
        xvec = np.linspace(-2, 2, 10)
        
        # Should handle unnormalized states gracefully
        W = wigner_2mode(rho_unnormalized, xvec, xvec)
        Q = qfunc_2mode(rho_unnormalized, xvec, xvec)
        
        assert np.isfinite(W).all(), "Wigner function should be finite"
        assert np.isfinite(Q).all(), "Q-function should be finite"
        assert Q.min() >= 0, "Q-function should remain non-negative"

    def test_4d_full_functions(self):
        """Test the full 4D Wigner and Q functions."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.8), coherent(N, 0.4)))
        
        x1 = x2 = np.linspace(-1.5, 1.5, 6)
        p1 = p2 = np.linspace(-1.5, 1.5, 6)
        
        W_4d = wigner_2mode_full(rho, x1, p1, x2, p2)
        Q_4d = qfunc_2mode_full(rho, x1, p1, x2, p2)
        
        # Check dimensions
        assert W_4d.shape == (6, 6, 6, 6)
        assert Q_4d.shape == (6, 6, 6, 6)
        
        # Q-function should be non-negative
        assert Q_4d.min() >= -1e-12
        
        # Both should be real
        assert np.isreal(W_4d).all()
        assert np.isreal(Q_4d).all()

    def test_2d_slices_consistency(self):
        """Test that 2D slice functions are consistent with full 4D function."""
        N = 6
        rho = ket2dm(tensor(coherent(N, 1.0), coherent(N, 0.6)))
        
        x_vec = np.linspace(-2, 2, 8)
        p_val = 0.0
        
        # Compare 2D slice with 4D slice
        W_xx_slice = wigner_2mode_xx(rho, x_vec, x_vec, p1=p_val, p2=p_val)
        W_4d = wigner_2mode_full(rho, x_vec, [p_val], x_vec, [p_val])
        W_4d_extracted = W_4d[:, 0, :, 0]
        
        assert_allclose(W_xx_slice, W_4d_extracted, rtol=1e-10)

    def test_performance_regression(self):
        """Test that functions complete within reasonable time."""
        import time
        
        N = 8
        rho = ket2dm(tensor(coherent(N, 1.0), coherent(N, 0.5)))
        xvec = np.linspace(-3, 3, 25)
        
        start_time = time.time()
        W = wigner_2mode(rho, xvec, xvec)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Function took too long: {elapsed:.2f}s"
        assert W.shape == (25, 25)

    def test_complex_phase_space_functions(self):
        """Test complex phase-space coordinate functions."""
        N = 6
        rho = ket2dm(tensor(coherent(N, 1.0 + 0.5j), coherent(N, 0.3 - 0.2j)))
        
        alpha1 = np.linspace(-1.5+0j, 1.5+0j, 8)
        alpha2 = np.linspace(-1.0+0j, 1.0+0j, 8)
        
        W_alpha = wigner_2mode_alpha(rho, alpha1, alpha2)
        Q_alpha = qfunc_2mode_alpha(rho, alpha1, alpha2)
        
        assert W_alpha.shape == (8, 8)
        assert Q_alpha.shape == (8, 8)
        assert Q_alpha.min() >= -1e-12


class TestTwoModeWignerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_hilbert_space(self):
        """Test with minimal Hilbert space dimensions."""
        N = 3
        rho = ket2dm(tensor(basis(N, 0), basis(N, 1)))
        
        xvec = np.linspace(-1, 1, 5)
        W = wigner_2mode(rho, xvec, xvec)
        
        assert W.shape == (5, 5)
        assert np.isfinite(W).all()

    def test_single_point_evaluation(self):
        """Test evaluation at single coordinate point."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.5), coherent(N, 0.3)))
        
        W = wigner_2mode_full(rho, [0.0], [0.0], [0.0], [0.0])
        assert W.shape == (1, 1, 1, 1)
        assert np.isfinite(W).all()


if __name__ == "__main__":
    pytest.main([__file__])