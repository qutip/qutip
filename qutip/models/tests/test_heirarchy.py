"""
Tests for the Heirarchy model
"""
from __future__ import division

from numpy.testing import 
import numpy as np

from qutip.models.heirarchy import Heirarchy
from qutip import Qobj

from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)
from scipy.integrate import quad, IntegrationWarning
from qutip import Qobj, sigmaz, basis, expect


class TestHeirarchy(object):
	"""
	Tests for the Heirarchy equation of motion model
	"""
	def test_rhs(self):
		"""
		Tests the generation of the RHS for the evolution of an initial
		density matrix using the Heirarchy equations of motion.
		"""
		Nc = 15
		wc = 0.05
		wq = 1.0

		lamJian = 0.01
		kappaJian=0.025

		ckAR = [lamJian/2.,lamJian/2.]
		ckAI = [-1.0j*lamJian/2.,1.0j*lamJian/2.]
		vkAR=[kappaJian+1.0j,kappaJian-1.0j]
		vkAI=[kappaJian+1.0j,kappaJian-1.0j]