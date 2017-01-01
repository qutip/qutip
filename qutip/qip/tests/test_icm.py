"""
Tests for the ICM module.
"""

import numpy as np
from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.qip.circuit import Gate
from qutip.qip.icm import Icm
from qutip import QubitCircuit

def test_decomposition():
	"""
	Test if initial decomposition in terms of T, V, P and CNOT is
	correct.
	"""
	qcirc = QubitCircuit(4, reverse_states=False)
	qcirc.add_gate("TOFFOLI", controls=[1, 2], targets=[0])
	model = Icm(qcirc)
	decomposed = model.decompose_gates()
	decomposed_gates = set([gate.name for gate in decomposed.gates])

	icm_gate_set = ["CNOT", "T", "V", "P", "T_dagger", "V_dagger", "P_dagger"]
	for gate in decomposed_gates:
		assert_(gate in icm_gate_set)

if __name__ == "__main__":
    run_module_suite()