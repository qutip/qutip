# qutip/tests/test_wigner_2mode.py (combined file)

"""
Comprehensive tests for two-mode Wigner and Q-function implementation.
Includes unit tests, integration tests, and regression protection.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from qutip import coherent, basis, ket2dm, tensor, wigner, squeeze
from qutip.wigner import (wigner_2mode, qfunc_2mode, wigner_2mode_full, 
                         qfunc_2mode_full, wigner_2mode_xx, wigner_2mode_xp,
                         wigner_2mode_alpha, qfunc_2mode_alpha)


# =============================================================================
# CORE FUNCTIONALITY TESTS
# =============================================================================

class TestTwoModeBasics:
    """Basic functionality and correctness tests."""
    
    def test_coherent_states_positive(self):
        """Coherent states should have predominantly positive Wigner function."""
        N = 8
        psi = tensor(coherent(N, 1.0), coherent(N, 0.5))
        rho = ket2dm(psi)
        
        xvec = np.linspace(-3, 3, 16)
        W = wigner_2mode(rho, xvec, xvec, interpretation='xx')
        
        assert W.shape == (16, 16)
        positive_fraction = np.sum(W > 0) / W.size
        assert positive_fraction > 0.5
        assert W.max() > 0.01
        assert W.min() > -0.5

    def test_squeezed_states_nonclassical(self):
        """Squeezed states show non-classical (negative) Wigner values."""
        N = 10
        psi = tensor(squeeze(N, 0.6) * coherent(N, 1.0), coherent(N, 0.0))
        rho = ket2dm(psi)
        
        xvec = np.linspace(-4, 4, 20)
        W = wigner_2mode(rho, xvec, xvec)
        
        assert W.min() < -1e-6, "Squeezed states should have negative Wigner values"
        assert W.max() > 0

    def test_vacuum_state(self):
        """Vacuum state should be positive and centered at origin."""
        N = 8
        vacuum = tensor(basis(N, 0), basis(N, 0))
        rho = ket2dm(vacuum)
        
        xvec = np.linspace(-2, 2, 12)
        W = wigner_2mode(rho, xvec, xvec)
        
        assert W.min() >= -1e-12, "Vacuum Wigner should be non-negative"
        max_idx = np.unravel_index(W.argmax(), W.shape)
        center_idx = (W.shape[0] // 2, W.shape[1] // 2)
        assert abs(max_idx[0] - center_idx[0]) <= 2
        assert abs(max_idx[1] - center_idx[1]) <= 2

    def test_qfunction_always_positive(self):
        """Q-function must always be non-negative."""
        N = 8
        states = [
            tensor(coherent(N, 1.0), coherent(N, 0.5)),
            tensor(squeeze(N, 0.3) * coherent(N, 0.8), coherent(N, 0.0))
        ]
        
        xvec = np.linspace(-3, 3, 12)
        for psi in states:
            rho = ket2dm(psi)
            Q = qfunc_2mode(rho, xvec, xvec)
            assert Q.min() >= -1e-12
            assert Q.max() > 0


# =============================================================================
# INTEGRATION AND NORMALIZATION TESTS
# =============================================================================

class TestIntegrationAndNormalization:
    """Tests validating integration behavior and normalization."""
    
    def test_vacuum_integration_to_single_mode(self):
        """∫∫ W_2mode(x1,p1,x2,p2) dx2 dp2 = W_1mode(x1,p1) for vacuum."""
        N = 10
        vacuum_2mode = ket2dm(tensor(basis(N, 0), basis(N, 0)))
        vacuum_1mode = ket2dm(basis(N, 0))
        
        x_range = np.linspace(-2, 2, 15)
        p_range = np.linspace(-2, 2, 15)
        
        W_2mode = wigner_2mode_full(vacuum_2mode, x_range, p_range, x_range, p_range)
        
        dx = x_range[1] - x_range[0]
        dp = p_range[1] - p_range[0]
        
        try:
            W_integrated = np.trapezoid(np.trapezoid(W_2mode, axis=3, dx=dp), axis=2, dx=dx)
        except AttributeError:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                W_integrated = np.trapz(np.trapz(W_2mode, axis=3, dx=dp), axis=2, dx=dx)
        
        W_1mode_ref = wigner(vacuum_1mode, x_range, p_range)
        
        assert_allclose(W_integrated, W_1mode_ref, rtol=1e-2, atol=1e-3)

    def test_vacuum_normalization_at_origin(self):
        """Vacuum state at origin should give W(0,0,0,0) ≈ 1/π² ≈ 0.101."""
        N = 8
        vacuum = tensor(basis(N, 0), basis(N, 0))
        rho = ket2dm(vacuum)
        
        W_origin = wigner_2mode_full(rho, [0.0], [0.0], [0.0], [0.0])
        origin_value = W_origin[0, 0, 0, 0]
        
        expected = 1.0 / (np.pi**2)  # ≈ 0.101321
        
        # Stricter test as requested by reviewer
        assert_allclose(origin_value, expected, rtol=0.02, atol=0.005,
                       err_msg=f"Expected {expected:.6f}, got {origin_value:.6f}")

    def test_coherent_state_maximum_values(self):
        """Displaced coherent states should maintain correct maximum W ≈ 1/π²."""
        N = 16
        g = np.sqrt(2)
        
        for alpha in [0.5, 1.0, 2.0]:
            rho = tensor(coherent(N, alpha), coherent(N, 0)).proj()
            
            x_center = 2 * np.real(alpha) / g
            p_center = 2 * np.imag(alpha) / g
            
            W_val = wigner_2mode_full(
                rho, [x_center], [p_center], [0.0], [0.0], method="optimized"
            )[0, 0, 0, 0]
            
            expected = 1.0 / (np.pi**2)
            assert np.isclose(W_val, expected, rtol=1e-2)

    def test_displaced_coherent_integration(self):
        """Integration of displaced two-mode coherent matches single-mode."""
        N = 20
        alpha = 1.0
        
        rho_2mode = ket2dm(tensor(coherent(N, alpha), basis(N, 0)))
        rho_1mode = ket2dm(coherent(N, alpha))
        
        x_range = np.linspace(-4, 4, 41)
        p_range = np.linspace(-4, 4, 41)
        
        W_2mode = wigner_2mode_full(rho_2mode, x_range, p_range, x_range, p_range)
        
        dx = x_range[1] - x_range[0]
        dp = p_range[1] - p_range[0]
        
        try:
            W_reduced = np.trapezoid(np.trapezoid(W_2mode, axis=3, dx=dp), axis=2, dx=dx)
        except AttributeError:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                W_reduced = np.trapz(np.trapz(W_2mode, axis=3, dx=dp), axis=2, dx=dx)
        
        W_single = wigner(rho_1mode, x_range, p_range)
        
        # Transpose required due to indexing='ij' convention in QuTiP vs 'xy' in matplotlib
        assert_allclose(W_reduced.T, W_single, rtol=1e-2, atol=1e-3)


# =============================================================================
# INPUT VALIDATION AND TYPE HANDLING
# =============================================================================

class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.5), coherent(N, 0.3)))
        xvec = np.linspace(-1, 1, 8)
        
        with pytest.raises(ValueError, match="method must be"):
            wigner_2mode_full(rho, xvec, xvec, xvec, xvec, method='invalid')

    def test_invalid_interpretation(self):
        """Invalid interpretation should raise ValueError."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.5), coherent(N, 0.3)))
        xvec = np.linspace(-1, 1, 8)
        
        with pytest.raises(ValueError, match="interpretation must be one of"):
            wigner_2mode(rho, xvec, xvec, interpretation='invalid')

    def test_single_mode_rejection(self):
        """Single-mode state should be rejected."""
        N = 5
        rho_1mode = ket2dm(coherent(N, 0.5))
        xvec = np.linspace(-1, 1, 8)
        
        with pytest.raises(ValueError, match="two-mode tensor operator"):
            wigner_2mode(rho_1mode, xvec, xvec)

    def test_automatic_ket_conversion(self):
        """Kets and bras should automatically convert to density matrices."""
        N = 6
        psi = tensor(coherent(N, 1.0), coherent(N, 0.5))
        rho = ket2dm(psi)
        
        xvec = np.linspace(-2, 2, 8)
        
        W_ket = wigner_2mode_full(psi, xvec, xvec, xvec, xvec)
        W_bra = wigner_2mode_full(psi.dag(), xvec, xvec, xvec, xvec)
        W_dm = wigner_2mode_full(rho, xvec, xvec, xvec, xvec)
        
        assert_allclose(W_ket, W_dm, rtol=1e-10)
        assert_allclose(W_bra, W_dm, rtol=1e-10)


# =============================================================================
# COORDINATE INTERPRETATIONS AND API
# =============================================================================

class TestCoordinateInterpretations:
    """Test different coordinate interpretation modes."""
    
    def test_all_interpretations_work(self):
        """Test xx, xp, and alpha interpretations."""
        N = 6
        rho = ket2dm(tensor(coherent(N, 0.8), coherent(N, 0.4)))
        coords = np.linspace(-2, 2, 10)
        
        # Test position-position and position-momentum slices
        W_xx = wigner_2mode(rho, coords, coords, interpretation='xx')
        W_xp = wigner_2mode(rho, coords, coords, interpretation='xp')
        
        assert W_xx.shape == (10, 10)
        assert W_xp.shape == (10, 10)
        
        # Different interpretations should give different results
        assert not np.allclose(W_xx, W_xp, rtol=0.1)
        
        # Test alpha interface with displacement method (returns proper 2D)
        alpha1 = np.linspace(-1.0, 1.0, 10) + 0j
        alpha2 = np.linspace(-0.8, 0.8, 10) + 0j
        W_alpha = wigner_2mode_alpha(rho, alpha1, alpha2, method='displacement')
        
        assert W_alpha.shape == (10, 10)
        assert np.isfinite(W_alpha).all()

    def test_4d_full_functions(self):
        """Test full 4D Wigner and Q functions."""
        N = 5
        rho = ket2dm(tensor(coherent(N, 0.8), coherent(N, 0.4)))
        
        x1 = x2 = np.linspace(-1.5, 1.5, 6)
        p1 = p2 = np.linspace(-1.5, 1.5, 6)
        
        W_4d = wigner_2mode_full(rho, x1, p1, x2, p2)
        Q_4d = qfunc_2mode_full(rho, x1, p1, x2, p2)
        
        assert W_4d.shape == (6, 6, 6, 6)
        assert Q_4d.shape == (6, 6, 6, 6)
        assert Q_4d.min() >= -1e-12
        assert np.isreal(W_4d).all()
        assert np.isreal(Q_4d).all()

    def test_g_parameter_consistency(self):
        """Test that g parameter works across all functions."""
        N = 6
        rho = ket2dm(tensor(coherent(N, 0.8), coherent(N, 0.4)))
        
        xvec = np.linspace(-2, 2, 8)
        g_test = 1.5
        
        W_2mode = wigner_2mode(rho, xvec, xvec, g=g_test)
        Q_2mode = qfunc_2mode(rho, xvec, xvec, g=g_test)
        W_xx = wigner_2mode_xx(rho, xvec, xvec, g=g_test)
        W_xp = wigner_2mode_xp(rho, xvec, xvec, g=g_test)
        
        assert np.isfinite(W_2mode).all()
        assert np.isfinite(Q_2mode).all()
        assert np.isfinite(W_xx).all()
        assert np.isfinite(W_xp).all()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
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

    def test_sparse_state(self):
        """Test that sparse states work correctly."""
        N = 12
        psi = (0.8 * tensor(basis(N, 0), basis(N, 0)) + 
               0.6 * tensor(basis(N, 1), basis(N, 0)))
        psi = psi.unit()
        rho = ket2dm(psi)
        
        coords = np.linspace(-1, 1, 6)
        W_opt = wigner_2mode_full(rho, coords, coords, coords, coords, method='optimized')
        
        assert np.isfinite(W_opt).all()
        assert W_opt.shape == (6, 6, 6, 6)


# =============================================================================
# PERFORMANCE (quick benchmarks only)
# =============================================================================

class TestPerformance:
    """Quick performance checks (not exhaustive benchmarks)."""
    
    def test_optimized_method_completes_quickly(self):
        """Optimized method should complete in reasonable time."""
        import time
        
        N = 8
        rho = ket2dm(tensor(coherent(N, 1.0), coherent(N, 0.5)))
        x_vec = np.linspace(-1, 1, 8)
        p_vec = np.linspace(-1, 1, 8)
        
        start = time.time()
        W_opt = wigner_2mode_full(rho, x_vec, p_vec, x_vec, p_vec, method='optimized')
        elapsed = time.time() - start
        
        assert np.isfinite(W_opt).all()
        assert elapsed < 5.0, f"Optimized method took {elapsed:.2f}s, should be <5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])