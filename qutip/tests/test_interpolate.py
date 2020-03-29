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


# We make a bunch of different test cases for the coefficients, all as a
# three-tuple of
#   (string, other, type)
# where `string` is the string-format time-dependence (we use this as the
# reference), `other` is what will become the time-dependence for this part of
# the test, and `type` is a string in ["spline", "function", "string"] which
# tells us how to interpret the type of `other`.  If `type` is "function" or
# "string", then nothing is done to `other` and it is used as-is.  If `type` is
# "spline", then `other` is given as a vectorised function which we turn into a
# qutip.Cubic_Spline using appropriate `x` and `y` values.
_case_real_spline = ('0.25*sin(t)', lambda t: 0.25*np.sin(t), 'spline')
_case_real_string = ('0.25*cos(t)', '0.25*cos(t)', 'string')
_case_real_function = ('0.25*sin(t)', lambda t, *_: 0.25*np.sin(t), 'function')
_case_complex_spline = ('0.1j*sin(t)', lambda t: 0.1j*np.sin(t), 'spline')
_case_complex_string = ('-0.1j*sin(t)', '-0.1j*sin(t)', 'string')


def _parametrize_solver_coefficients(metafunc):
    """
    Perform the parametrisation for test cases using a solver and some
    time-dependent coefficients.  This is necessary because not all solvers can
    accept all combinations of time-dependence specifications.
    """
    solvers = [
        (qutip.sesolve, 'sesolve'),
        (qutip.mesolve, 'mesolve'),
        (functools.partial(qutip.mcsolve, ntraj=500), "mcsolve"),
        (qutip.brmesolve, 'brmesolve')
    ]
    coefficients = [
        ([_case_real_spline], "real-spline"),
        ([_case_real_spline, _case_real_string], "real-spline,string"),
        ([_case_real_spline, _case_real_function], "real-spline,function"),
        ([_case_real_function], "real-function"),
        ([_case_complex_spline, _case_complex_string],
            "complex-spline,string"),
    ]

    def _valid_case(solver_id, coefficient_id):
        invalids = [
            "brmesolve" in solver_id and "function" in coefficient_id,
        ]
        return not any(invalids)

    def _make_case(solver, solver_id, coefficient, coefficient_id):
        marks = []
        if "brmesolve" in solver_id:
            marks.append(pytest.mark.requires_cython)
        id_ = "-".join([solver_id, coefficient_id])
        return pytest.param(solver, coefficient, id=id_, marks=marks)

    cases = [_make_case(solver, solver_id, coefficient, coefficient_id)
             for solver, solver_id in solvers
             for coefficient, coefficient_id in coefficients
             if _valid_case(solver_id, coefficient_id)]
    metafunc.parametrize(["solver", "coefficients"], cases)


@pytest.mark.slow
def test_usage_in_solvers(solver, coefficients):
    """
    Test that the Cubic_Spline can be used as a time-dependent argument to all
    three principle solvers, both alone and in combination with other forms of
    time-dependence.
    """
    size = 10
    a = qutip.destroy(size)
    state = qutip.basis(size, 1)
    times = np.linspace(0, 5, 50)
    e_ops = [a.dag() * a]
    H_expected = [a.dag()*a]
    H_test = H_expected.copy()
    operator = a**2 + a.dag()**2
    for string, other, type_ in coefficients:
        if type_ == 'spline':
            other = qutip.Cubic_Spline(times[0], times[-1], other(times))
        H_expected.append([operator, string])
        H_test.append([operator, other])
    expected = solver(H_expected, state, times, e_ops=e_ops).expect[0]
    test = solver(H_test, state, times, e_ops=e_ops).expect[0]
    np.testing.assert_allclose(test, expected, atol=1e-4)
