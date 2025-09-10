import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from qutip import (sigmaz, sigmax, basis, mesolve, mcsolve, 
                   Options, Result)
# To this (use the same pattern as in your result.py):
try:
    import matplotlib
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

@pytest.mark.skipif(not matplotlib_available, reason="matplotlib not available")
class TestResultPlot:
    
    def test_standard_result_plot(self):
        """Test plotting for standard results."""
        # Simple system
        H = sigmaz()
        psi0 = basis(2, 0)
        times = np.linspace(0, 10, 100)
        e_ops = [sigmax(), sigmaz()]
        
        result = mesolve(H, psi0, times, [], e_ops)
        
        # Test basic plotting
        fig, ax = result.plot_expect(show=False)
        assert fig is not None
        assert ax is not None
        
        # Test with dictionary e_ops
        e_ops_dict = {'sx': sigmax(), 'sz': sigmaz()}
        result_dict = mesolve(H, psi0, times, [], e_ops_dict)
        fig, ax = result_dict.plot_expect(show=False)
        assert fig is not None
        assert ax is not None
    
    def test_mcsolve_result_plot(self):
        """Test plotting for mcsolve results."""
        H = sigmaz()
        psi0 = basis(2, 0)
        times = np.linspace(0, 5, 50)
        e_ops = [sigmax()]
        
        # Run with few trajectories for speed
        options = Options(nsteps=1000, store_states=True)
        result = mcsolve(H, psi0, times, [], e_ops, ntraj=3, options=options)
        
        # Test average plotting
        fig, ax = result.plot_expect(average=True, show=False)
        assert fig is not None
        assert ax is not None
        
        # Test trajectories plotting
        fig, ax = result.plot_expect(trajectories=2, average=False, show=False)
        assert fig is not None
        assert ax is not None
    
    def test_plot_with_fig_axes(self):
        """Test plotting with provided fig and axes."""
        H = sigmaz()
        psi0 = basis(2, 0)
        times = np.linspace(0, 10, 100)
        e_ops = [sigmax()]
        
        result = mesolve(H, psi0, times, [], e_ops)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        fig, ax = result.plot_expect(fig=fig, axes=ax, show=False)
        assert fig is not None
        assert ax is not None
    
    def test_plot_with_kwargs(self):
        """Test plotting with matplotlib kwargs."""
        H = sigmaz()
        psi0 = basis(2, 0)
        times = np.linspace(0, 10, 100)
        e_ops = [sigmax()]
        
        result = mesolve(H, psi0, times, [], e_ops)
        
        fig, ax = result.plot_expect(show=False, color='red', linestyle='--', linewidth=2)
        assert fig is not None
        assert ax is not None