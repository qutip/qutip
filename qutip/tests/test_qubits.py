from numpy.testing import assert_, run_module_suite
from qutip.qip.qubits import qubit_states
from qutip.tensor import tensor
from qutip.states import basis


class TestQubits:
    """
    A test class for the QuTiP functions for qubits.
    """
    def testQubitStates(self):
        """
        Tests the qubit_states function.
        """
        psi0_a = basis(2, 0)
        psi0_b = qubit_states()
        assert_(psi0_a == psi0_b)

        psi1_a = basis(2, 1)
        psi1_b = qubit_states(states=[1])
        assert_(psi1_a == psi1_b)

        psi01_a = tensor(psi0_a, psi1_a)
        psi01_b = qubit_states(N=2, states=[0, 1])
        assert_(psi01_a == psi01_b)


if __name__ == "__main__":
    run_module_suite()
