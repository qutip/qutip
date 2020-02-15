# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import pytest
import collections
import functools
import numpy as np
import qutip

_Case = collections.namedtuple('_Case', ['operator', 'state', 'expected'])


def _case_to_dm(case):
    return case._replace(state=[x.proj() for x in case.state])


def _unwrap(list_):
    """Unwrap lists until we reach the first non-list element."""
    out = list_
    while isinstance(out, list):
        out = out[0]
    return out


def _case_id(case):
    op_part = 'qubit' if _unwrap(case.operator).dims[0][0] == 2 else 'basis'
    state_part = 'ket' if _unwrap(case.state).dims[1][0] == 1 else 'dm'
    return op_part + "-" + state_part


# Store cases for the tests of known expectation value in broadcasted form,
# because it's easier to unroll broadcasted objects than roll up flat ones.
_dim = 5
_num, _a = qutip.num(_dim), qutip.destroy(_dim)
_sx, _sz, _sp = qutip.sigmax(), qutip.sigmaz(), qutip.sigmap()
_known_fock = _Case([_num, _a],
                    [qutip.fock(_dim, n) for n in range(_dim)],
                    np.array([np.arange(_dim), np.zeros(_dim)]))
_known_qubit = _Case([_sx, _sz, _sp],
                     [qutip.basis(2, 0), qutip.basis(2, 1)],
                     np.array([[0, 0], [1, -1], [0, 0]]))
_known_cases = [_known_fock, _case_to_dm(_known_fock),
                _known_qubit, _case_to_dm(_known_qubit)]


class TestKnownExpectation:
    def pytest_generate_tests(self, metafunc):
        cases = _known_cases
        op_name, state_name = 'operator', 'state'
        if op_name not in metafunc.fixturenames:
            op_name += 's'
        else:
            cases = [_Case(op, case.state, expected)
                     for case in cases
                     for op, expected in zip(case.operator, case.expected)]
        if state_name not in metafunc.fixturenames:
            state_name += 's'
        else:
            cases = [_Case(case.operator, state, expected)
                     for case in cases
                     for state, expected in zip(case.state, case.expected.T)]
        metafunc.parametrize([op_name, state_name, 'expected'], cases,
                             ids=[_case_id(case) for case in cases])

    def test_operator_by_basis(self, operator, state, expected):
        result = qutip.expect(operator, state)
        assert result == expected
        assert isinstance(result, float if operator.isherm else complex)

    def test_broadcast_operator_list(self, operators, state, expected):
        result = qutip.expect(operators, state)
        expected_dtype = (np.float64 if all(op.isherm for op in operators)
                          else np.complex128)
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected_dtype
        assert list(result) == list(expected)

    def test_broadcast_state_list(self, operator, states, expected):
        result = qutip.expect(operator, states)
        expected_dtype = np.float64 if operator.isherm else np.complex128
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected_dtype
        assert list(result) == list(expected)

    def test_broadcast_both_lists(self, operators, states, expected):
        result = qutip.expect(operators, states)
        assert len(result) == len(operators)
        for part, operator, expected_part in zip(result, operators, expected):
            expected_dtype = np.float64 if operator.isherm else np.complex128
            assert isinstance(part, np.ndarray)
            assert part.dtype == expected_dtype
            assert list(part) == list(expected_part)


@pytest.mark.repeat(20)
@pytest.mark.parametrize("hermitian", [False, True], ids=['complex', 'real'])
def test_equivalent_to_matrix_element(hermitian):
    dimension = 20
    state = qutip.rand_ket(dimension, 0.3)
    op = qutip.rand_herm(dimension, 0.2)
    if not hermitian:
        op = op + 1j*qutip.rand_herm(dimension, 0.1)
    matrix = (state.dag() * op * state).data[0, 0]
    assert abs(qutip.expect(op, state) - matrix) < 1e-14


@pytest.mark.parametrize("solve", [
    pytest.param(qutip.sesolve, id="sesolve"),
    pytest.param(functools.partial(qutip.mesolve, c_ops=[0.001 * _sz]),
                 id="mesolve"),
])
def test_compatibility_with_solver(solve):
    e_ops = [getattr(qutip, 'sigma'+x)() for x in 'xyzmp']
    h = qutip.sigmax()
    state = qutip.basis(2, 0)
    times = np.linspace(0, 10, 101)
    direct = solve(h, state, times, e_ops=e_ops).expect
    indirect = qutip.expect(e_ops, solve(h, state, times).states)
    assert len(direct) == len(indirect)
    for direct_, indirect_ in zip(direct, indirect):
        assert len(direct_) == len(indirect_)
        assert isinstance(direct_, np.ndarray)
        assert isinstance(indirect_, np.ndarray)
        assert direct_.dtype == indirect_.dtype
        assert np.allclose(direct_, indirect_, atol=1e-12)
