"""
Tests for the Hierarchy model
"""
from __future__ import division

import numpy as np
from numpy.testing import (assert_, assert_almost_equal,
                           run_module_suite, assert_equal,
                           assert_array_equal, assert_array_almost_equal)

from scipy.integrate import quad, IntegrationWarning

from qutip import Qobj, sigmaz, sigmax, basis, Options
from qutip.models.hierarchy import Heom, _heom_state_dictionaries


class TestHeom(object):
    """
    Tests for the Heirarchy equation of motion model
    """
    def test_rhs(self):
        """Tests the generation of the RHS for the evolution of an initial
        density matrix using the Hierarchy equations of motion.
        """
        lam = 0.01
        kappa =0.025

        ckAR = [lam/2., lam/2.]
        ckAI = [-1.0j*lam/2.,1.0j*lam/2.]
        vkAR=[kappa+1.0j,kappa -1.0j]
        vkAI=[kappa+1.0j,kappa-1.0j]


        Q=sigmax()
        Del = 0.   
        wq = 1.0     # Energy of the 2-level system.
        Hsys = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

        NR = len(ckAR)
        NI = len(ckAI)
        wc = 0.05
        Nc = 1

        Hsys = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()


        system = Heom(Hsys, Q, Nc = Nc, real_coeff=ckAR, real_freq=vkAR,
                           imaginary_coeff=ckAI, imaginary_freq=vkAI)

        rhs_calculated = system._rhs(progress_bar=True)
        row0_expected = np.array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
                                  0.+0.j,0.-0.07071068j,0.+0.07071068j,
                                  0.+0.j, 0.+0.j,0.-0.07071068j,
                                  0.+0.07071068j,0.+0.j,
                                  0.+0.j,0.-0.07071068j,0.+0.07071068j,
                                  0.+0.j, 0.+0.j,0.-0.07071068j,
                                  0.+0.07071068j,0.+0.j])

        assert_array_almost_equal(rhs_calculated.toarray()[0], row0_expected)
        
        rhoend = np.array([0.000+0.j,0.000+0.07071068j,0.000-0.07071068j,
                         0.000+0.j,0.000+0.j,0.000+0.31213203j,
                         0.000-0.31213203j,0.000+0.j,0.000+0.j,
                         0.000+0.j,0.000+0.j,0.000+0.j,
                         0.000+0.j,0.000+0.j,0.000+0.j,
                         0.000+0.j,0.000+0.j,0.000+0.j,
                         0.000+0.j,-0.025-1.j])

        assert_array_almost_equal(rhs_calculated.toarray()[-1], rhoend)

        diagonal = np.array([ 0.000+0.j,0.000+1.j,0.000-1.j,0.000+0.j,
                             -0.025+1.j, -0.025+2.j, -0.025+0.j, -0.025+1.j,
                             -0.025-1.j, -0.025+0.j, -0.025-2.j, -0.025-1.j,
                             -0.025+1.j, -0.025+2.j, -0.025+0.j, -0.025+1.j,
                             -0.025-1.j, -0.025+0.j, -0.025-2.j, -0.025-1.j])

        assert_array_almost_equal(np.diagonal(rhs_calculated.toarray()),
                              diagonal)

    def test_solve(self):
        """
        Test the evolution of the density matrix using the solver
        """
        lam = 0.01
        kappa =0.025

        ckAR = [lam/2., lam/2.]
        ckAI = [-1.0j*lam/2.,1.0j*lam/2.]
        vkAR=[kappa+1.0j,kappa -1.0j]
        vkAI=[kappa+1.0j,kappa-1.0j]


        Q=sigmax()
        Del = 0.   
        wq = 1.0     # Energy of the 2-level system.
        Hsys = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

        NR = len(ckAR)
        NI = len(ckAI)
        wc = 0.05
        Nc = 2

        Hsys = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()


        system = Heom(Hsys, Q, Nc=Nc, real_coeff=ckAR,real_freq=vkAR,
                           imaginary_coeff=ckAI, imaginary_freq=vkAI)
        
        wc = 0.05                # Cutoff frequency.

        tlist = np.linspace(0, 5, 100)
        initial_state = basis(2,0) * basis(2,0).dag() # Initial state

        options = Options(nsteps=1500, store_states=True, atol=1e-12,
                          rtol=1e-12)

        result = system.solve(initial_state, tlist, options)
        final_state = result.states[-1].full()
        norm = np.linalg.norm(final_state)

        assert_equal(norm, 0.81087653436037588)

    def test_heom_state_dictionaries(self):
        """
        Test the HEOM state and index generation function
        """
        nstates, state2idx, idx2state = _heom_state_dictionaries([2, 2, 2], 1)
        assert_equal(nstates, 4)
        assert_equal(state2idx, {(0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2,
                                 (1, 0, 0): 3})
        assert_equal(idx2state, {0: (0, 0, 0), 1: (0, 0, 1),
                                 2: (0, 1, 0), 3: (1, 0, 0)})

