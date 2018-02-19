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
from qutip.models.hierarchy import Heom, add_at_idx


class TestHeom(object):
    """
    Tests for the hierarchy equation of motion model
    """
    def test_initialization(self):
        """
        Tests the correct initialization of the model class
        and checks if the shapes of the rho term is constructed properly
        for all the density matrices.
        """
        coupling = sigmax()
        delta = 0.5
        wq = 0.5
        hamiltonian = 0.5 * wq * sigmaz() + 0.5 * delta * sigmax()
        hierarchy_levels = 2

        lam = .01
        kappa = 0.05

        ck = [(1-1.0j)*lam/2., (1+1.0j)*lam/2.]
        vk = [kappa + 1.0j, kappa - 1.0j]

        model = Heom(hamiltonian, coupling, ck, vk, ncut=hierarchy_levels)
        assert_equal(model.nhe, 6)
        assert_equal(model.idx2he[0], (0, 0))
        assert_equal(model.he2idx[(0, 0)], 0)
        assert_equal(model.N, 2)
        assert_equal(model.hshape, (6, 4))
        assert_equal(model.grad_shape, (4, 4))
        
        num_hierarchy = 3
        model = Heom(hamiltonian, coupling, ck, vk, ncut=num_hierarchy)
        assert_equal(model.nhe, 10)
        assert_equal(model.idx2he[0], (0, 0))
        assert_equal(model.he2idx[(0, 0)], 0)
        assert_equal(model.hshape, (10, 4))
        assert_equal(model.grad_shape, (4, 4))

        assert_equal((4, 4), model.L.shape) 
        
    def test_hierarchy_idx(self):
        """
        Test the HEOM state and index generation function
        """
        coupling = sigmax()
        delta = 0.5
        wq = 0.5
        hamiltonian = 0.5 * wq * sigmaz() + 0.5 * delta * sigmax()
        hierarchy_levels = 5

        lam = .01
        kappa = 0.05

        ck = [(1-1.0j)*lam/2., (1+1.0j)*lam/2., 0, 0]
        vk = [kappa + 1.0j, kappa - 1.0j, 0, 0]

        model = Heom(hamiltonian, coupling, ck, vk, ncut=hierarchy_levels)

        assert_equal(model.nhe, 126)
        assert_equal(model.idx2he[0], (0, 0, 0, 0))
        assert_((5, 0, 0, 0) in model.he2idx)
        assert_((2, 2, 1, 0) in model.he2idx)   

    def test_add_at_idx(self):
        """
        Test the adding of a number at an id
        """
        tup = (1, 0, 1, 4)
        assert_equal(add_at_idx(tup, 2, -1), (1, 0, 0, 4))
        assert_equal(add_at_idx(tup, 0, 1), (2, 0, 1, 4))
        assert_equal(tup, (1, 0, 1, 4))

    def test_prev(self):
        """
        Test the generation of the previous and next indices in the hierarchy
        """
        coupling = sigmax()
        delta = 0.5
        wq = 0.5
        hamiltonian = 0.5 * wq * sigmaz() + 0.5 * delta * sigmax()
        hierarchy_levels = 5

        lam = .01
        kappa = 0.05

        ck = [(1-1.0j)*lam/2., (1+1.0j)*lam/2., 0, 0]
        vk = [kappa + 1.0j, kappa - 1.0j, 0, 0]

        model = Heom(hamiltonian, coupling, ck, vk, ncut=hierarchy_levels)

        assert_equal(model.prev_next(0, 0), (np.nan, model.he2idx[1,0,0,0]))
        assert_equal(model.prev_next(model.he2idx[(0,0,5,0)], 2),
                     (model.he2idx[(0,0,4,0)], np.nan))
        assert_equal(model.prev_next(5, 2), (np.nan, np.nan))

    def test_norm(self):
        """
        Tests the normalization of the auxillary density matrices
        """
        coupling = sigmax()
        delta = 0.5
        wq = 0.5
        hamiltonian = 0.5 * wq * sigmaz() + 0.5 * delta * sigmax()
        hierarchy_levels = 5

        lam = .01
        kappa = 0.05

        ck = [(1-1.0j)*lam/2., (1+1.0j)*lam/2., 0, 0]
        vk = [kappa + 1.0j, kappa - 1.0j, 0, 0]

        model = Heom(hamiltonian, coupling, ck, vk, ncut=hierarchy_levels)
        assert_equal(model.norm((0, 0, 0, 0)), 1)
        assert_equal(model.norm((1, 1, 1, 1)), 0)

        ck = [(1-1.0j)*lam/2., (1+1.0j)*lam/2., 2, -2]
        vk = [kappa + 1.0j, kappa - 1.0j, 1, 1]

        model = Heom(hamiltonian, coupling, ck, vk, ncut=hierarchy_levels)
        assert_equal(model.norm((0, 0, 0, 0)), 1)
        assert_almost_equal(model.norm((1, 1, 1, 1)), 2*lam**2)
        assert_almost_equal(model.norm((0, 2, 0, 3)), 48*lam**2)

    def test_deltak(self):
        """
        Test the `_deltak` function which gives the factor to be used
        for the truncation of the Matsubara terms using the Ishizaki
        Tanimura scheme
        """
        pass

    def test_gradn(self):
        """
        Test the generation of the correct gradient value for the hierarchy at
        `n`.
        """
        pass

    def test_grad_prev_next(self):
        """
        Test the generation of the correct gradient value for the hierarchy at
        previous and next terms
        """
        pass


if __name__ == "__main__":
    np.run_module_suite()

