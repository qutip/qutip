__all__ = ['qubit_states']

from qutip.tensor import tensor
from numpy import sqrt
from qutip.states import basis


def qubit_states(N=1, states=[0]):
    """
    Function to define initial state of the qubits.

    Parameters
    ----------
    N : Integer
        Number of qubits in the register.
    states : List
        Initial state of each qubit.

    Returns
    ----------
    qstates : Qobj
        List of qubits.

    """
    state_list = []
    for i in range(N):
        if N > len(states) and i >= len(states):
            state_list.append(0)
        else:
            state_list.append(states[i])

    return tensor([alpha * basis(2, 1) + sqrt(1 - alpha**2) * basis(2, 0)
                  for alpha in state_list])
