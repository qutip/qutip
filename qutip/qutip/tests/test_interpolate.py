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
import functools
import numpy as np
import scipy.interpolate
import pytest
import qutip

pytestmark = [pytest.mark.usefixtures("in_temporary_directory")]


def pytest_generate_tests(metafunc):
    """
    Perform more complex parametrisation logic for tests in this module.  This
    was originally needed because the `brmesolve` solver cannot mix splines and
    functions in the Hamiltonian (it can only accept string-formatted
    time-dependence), but we want to parametrize this for all the other tests.
    """
    if ("solver" in metafunc.fixturenames
            and "coefficients" in metafunc.fixturenames):
        _parametrize_solver_coefficients(metafunc)


@pytest.mark.parametrize('noise', [
    pytest.param(lambda n: 0.1*np.random.rand(n), id="real"),
    pytest.param(lambda n: 0.1j*np.random.rand(n), id="complex"),
])
def test_equivalence_to_scipy(noise):
    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x) + noise(x.shape[0])
    test = qutip.Cubic_Spline(x[0], x[-1], y)
    expected = scipy.interpolate.CubicSpline(x, y, bc_type="natural")
    # We use the bc_type="natural", i.e. zero second-derivatives at the
    # boundaries because that matches qutip.
    np.testing.assert_allclose(test(x), expected(x), atol=1e-10)
    centres = 0.5 * (x[:-1] + x[1:])
    # Test some points not in the original.
    for point in centres:
        assert np.abs(test(point) - expected(point)) < 1e-10, "Scalar argument"


class _Case:
    """
    Represents a time-dependent coefficient test case for
    test_usage_in_solvers.  The functions `function`, `spline` and `string`
    return a tuple of (reference, test), where the reference is a function type
    and the test is the type of whatever was requested.  This lets us build up
    test cases where we can get the same coefficient in different formats, so
    the tests can be parametrised.
    """
    def __init__(self, amplitude, function, string):
        self.amplitude = amplitude
        self._function = function
        self.string_coefficient = string

    def __call__(self, t, *_):
        return self.amplitude * self._function(t)

    def function(self):
        return self, self

    def spline(self, times):
        return self, qutip.Cubic_Spline(times[0], times[-1], self(times))

    def string(self):
        return self, self.string_coefficient


_real_cos = _Case(0.25, np.cos, '0.25*cos(t)')
_real_sin = _Case(0.25, np.sin, '0.25*sin(t)')
_complex_pos = _Case(0.1j, np.sin, '0.1j*sin(t)')
_complex_neg = _Case(-0.1j, np.sin, '-0.1j*sin(t)')


def _parametrize_solver_coefficients(metafunc):
    """
    Perform the parametrisation for test cases using a solver and some
    time-dependent coefficients.  This is necessary because not all solvers can
    accept all combinations of time-dependence specifications.
    """
    size = 10
    times = np.linspace(0, 5, 50)
    c_ops = [qutip.qzero(size)]
    solvers = [
        (qutip.sesolve, 'sesolve'),
        (functools.partial(qutip.mesolve, c_ops=c_ops), 'mesolve'),
        (functools.partial(qutip.mcsolve, c_ops=c_ops), "mcsolve"),
        (qutip.brmesolve, 'brmesolve'),
    ]
    # A list of (reference, test) pairs, where the reference is in function
    # form because that's the fastest, and the test is named type.
    coefficients = [
        ([_real_cos.spline(times)], "real-spline"),
        ([_real_cos.string(), _real_sin.spline(times)], "real-string,spline"),
        ([_real_cos.spline(times), _real_sin.function()],
            "real-spline,function"),
        ([_complex_pos.spline(times), _complex_neg.string()],
            "complex-spline,string"),
    ]

    def _valid_case(solver_id, coefficient_id):
        invalids = [
            "brmesolve" in solver_id and "function" in coefficient_id,
        ]
        return not any(invalids)

    def _make_case(solver, solver_id, coefficients, coefficient_id):
        marks = []
        if "brmesolve" in solver_id:
            # We must use the string as the reference type instead of the
            # function if brmesolve is in use.
            marks.append(pytest.mark.requires_cython)
            coefficients = [(c.string_coefficient, other)
                            for c, other in coefficients]
        id_ = "-".join([solver_id, coefficient_id])
        return pytest.param(solver, coefficients, size, times,
                            id=id_, marks=marks)

    cases = [_make_case(solver, solver_id, coefficients_, coefficient_id)
             for solver, solver_id in solvers
             for coefficients_, coefficient_id in coefficients
             if _valid_case(solver_id, coefficient_id)]
    metafunc.parametrize(["solver", "coefficients", "size", "times"], cases)


@pytest.mark.slow
def test_usage_in_solvers(solver, coefficients, size, times):
    """
    Test that the Cubic_Spline can be used as a time-dependent argument to all
    three principle solvers, both alone and in combination with other forms of
    time-dependence.
    """
    a = qutip.destroy(size)
    state = qutip.basis(size, 1)
    e_ops = [a.dag() * a]
    H_expected = [a.dag()*a]
    H_test = H_expected.copy()
    operator = a**2 + a.dag()**2
    for expected, test in coefficients:
        H_expected.append([operator, expected])
        H_test.append([operator, test])
    expected = solver(H_expected, state, times, e_ops=e_ops).expect[0]
    test = solver(H_test, state, times, e_ops=e_ops).expect[0]
    np.testing.assert_allclose(test, expected, atol=1e-4)
