import numpy as np
from numpy.testing import assert_, assert_equal, run_module_suite
from qutip.states import basis
from qutip.three_level_atom import *


three_states = three_level_basis()
three_check = np.empty((3,), dtype=object)
three_check[:] = [basis(3, 0), basis(3, 1), basis(3, 2)]
three_ops = three_level_ops()


def testThreeStates():
    "Three-level atom: States"
    assert_equal(np.all(three_states == three_check), True)


def testThreeOps():
    "Three-level atom: Operators"
    assert_equal((three_ops[0]*three_states[0]).full(), three_check[0].full())
    assert_equal((three_ops[1]*three_states[1]).full(), three_check[1].full())
    assert_equal((three_ops[2]*three_states[2]).full(), three_check[2].full())
    assert_equal((three_ops[3]*three_states[1]).full(), three_check[0].full())
    assert_equal((three_ops[4]*three_states[1]).full(), three_check[2].full())

if __name__ == "__main__":
    run_module_suite()
