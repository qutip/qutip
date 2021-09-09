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
import pickle
import numpy as np
from numpy.testing import assert_, run_module_suite
from qutip import (
    sigmax, sigmay, sigmaz, qeye, basis, expect, num, destroy, create,
    Cubic_Spline, QobjEvo, Qobj
)
from qutip.solver.sesolve import sesolve, SeSolver
from qutip.solver.options import SolverOptions
from qutip.solver.solver_base import Solver
from qutip.solver.integrator import *

all_ode_method = Solver._avail_integrators.keys()


class TestSeSolve():
    H0 = 0.2 * np.pi * sigmaz()
    H1 = np.pi * sigmax()
    tlist = np.linspace(0, 20, 200)
    S = Cubic_Spline(0, 20, np.exp(-0.5 * tlist))
    args = {'alpha': 0.5}
    w_a = 0.35
    a = 0.5

    @pytest.mark.parametrize(['unitary_op'],
        [pytest.param(None, id="state"),
         pytest.param(qeye(2), id="unitary"),
    ])
    @pytest.mark.parametrize(['H', 'analytical'],
        [pytest.param(H1,
                      lambda t, args: t,
                      id='const_H'),
         pytest.param(lambda t, args: np.pi * sigmax() * np.exp(-args['alpha'] * t),
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='func_H'),
         pytest.param([[H1, lambda t, args: np.exp(-args['alpha'] * t)]],
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='list_func_H'),
         pytest.param([[H1, 'exp(-alpha*t)']],
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='list_str_H'),
         pytest.param([[H1, S]],
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='list_cubic_spline_H'),
         pytest.param([[H1, np.exp(-args['alpha'] * tlist)]],
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='list_array_H'),
         pytest.param(QobjEvo([[H1, 'exp(-alpha*t)']], args={'alpha': 0.5}),
                      lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                       / args['alpha']),
                      id='QobjEvo_H'),
    ])
    def test_sesolve(self, H, analytical, unitary_op, tol=5e-3):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """
        psi0 = basis(2, 0)
        option = SolverOptions(progress_bar=None)

        if unitary_op is None:
            output = sesolve(H, psi0, self.tlist,
                             [sigmax(), sigmay(), sigmaz()],
                             args=self.args)
            sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
        else:
            output = sesolve(H, unitary_op, self.tlist, args=self.args)
            sx = [expect(sigmax(), U * psi0) for U in output.states]
            sy = [expect(sigmay(), U * psi0) for U in output.states]
            sz = [expect(sigmaz(), U * psi0) for U in output.states]

        sx_analytic = np.zeros(np.shape(self.tlist))
        sy_analytic = np.array([-np.sin(2*np.pi * analytical(t, self.args))
                                for t in self.tlist])
        sz_analytic = np.array([np.cos(2*np.pi * analytical(t, self.args))
                                for t in self.tlist])

        np.testing.assert_allclose(sx, sx_analytic, atol=tol)
        np.testing.assert_allclose(sy, sy_analytic, atol=tol)
        np.testing.assert_allclose(sz, sz_analytic, atol=tol)

    @pytest.mark.parametrize(['unitary_op'],
        [pytest.param(None, id="state"),
         pytest.param(qeye(2), id="unitary"),
    ])
    @pytest.mark.parametrize('method', all_ode_method, ids=all_ode_method)
    def test_sesolve_method(self, method, unitary_op):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """
        tol = 5e-3
        psi0 = basis(2, 0)
        options = SolverOptions(method=method, progress_bar=None)
        H = [[self.H1, 'exp(-alpha*t)']]

        if unitary_op is None:
            output = sesolve(H, psi0, self.tlist,
                             [sigmax(), sigmay(), sigmaz()],
                             args=self.args, options=options)
            sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
        else:
            output = sesolve(H, unitary_op, self.tlist,
                             args=self.args, options=options)
            sx = [expect(sigmax(), U * psi0) for U in output.states]
            sy = [expect(sigmay(), U * psi0) for U in output.states]
            sz = [expect(sigmaz(), U * psi0) for U in output.states]

        sx_analytic = np.zeros(np.shape(self.tlist))
        sy_analytic = np.array([np.sin(-2*np.pi *
                                       ((1 - np.exp(-self.args['alpha'] * t)) /
                                        self.args['alpha']))
                                for t in self.tlist])
        sz_analytic = np.array([np.cos(2*np.pi *
                                       ((1 - np.exp(-self.args['alpha'] * t)) /
                                        self.args['alpha']))
                                for t in self.tlist])

        np.testing.assert_allclose(sx, sx_analytic, atol=tol)
        np.testing.assert_allclose(sy, sy_analytic, atol=tol)
        np.testing.assert_allclose(sz, sz_analytic, atol=tol)

    @pytest.mark.parametrize('normalize', [True, False], ids=['Normalized', ''])
    @pytest.mark.parametrize(['H', 'args'],
        [pytest.param(H0 + H1,
                      {},
                      id='const_H'),
         pytest.param(lambda t, args: args['a'] * t * 0.2 * np.pi * sigmaz() + \
                                      np.cos(args['w_a'] * t) * np.pi * sigmax(),
                      {'a':a, 'w_a':w_a},
                      id='func_H'),
         pytest.param([[H0, lambda t, args: args['a']*t],
                       [H1, lambda t, args: np.cos(args['w_a']*t)]],
                      {'a':a, 'w_a':w_a},
                      id='list_func_H'),
         pytest.param([H0, [H1, 'cos(w_a*t)']],
                      {'w_a':w_a},
                      id='list_str_H'),
    ])
    def test_compare_evolution(self, H, normalize, args, tol=5e-5):
        """
        Compare integrated evolution of unitary operator with state evo
        """
        psi0 = basis(2, 0)
        U0 = qeye(2)

        options = SolverOptions(store_states=True, normalize_output=normalize,
                                progress_bar=None)
        out_s = sesolve(H, psi0, self.tlist, [sigmax(), sigmay(), sigmaz()],
                        options=options, args=args)
        xs, ys, zs = out_s.expect[0], out_s.expect[1], out_s.expect[2]
        xss = [expect(sigmax(), U) for U in out_s.states]
        yss = [expect(sigmay(), U) for U in out_s.states]
        zss = [expect(sigmaz(), U) for U in out_s.states]

        assert (max(abs(xs - xss)) < tol)
        assert (max(abs(ys - yss)) < tol)
        assert (max(abs(zs - zss)) < tol)

        if normalize:
            # propagator evolution is not normalized (yet?)
            tol = 5e-4
        out_u = sesolve(H, U0, self.tlist, options=options, args=args)
        xu = [expect(sigmax(), U * psi0) for U in out_u.states]
        yu = [expect(sigmay(), U * psi0) for U in out_u.states]
        zu = [expect(sigmaz(), U * psi0) for U in out_u.states]

        assert (max(abs(xs - xu)) < tol)
        assert (max(abs(ys - yu)) < tol)
        assert (max(abs(zs - zu)) < tol)

    def test_sesolver_pickling(self):
        options = SolverOptions(progress_bar=None)
        solver_obj = SeSolver(self.H0 + self.H1,
                              e_ops=[sigmax(), sigmay(), sigmaz()],
                              options=options)
        copy = pickle.loads(pickle.dumps(solver_obj))
        sx, sy, sz = solver_obj.run(basis(2,1), [0, 1, 2, 3], {}).expect
        csx, csy, csz = solver_obj.run(basis(2,1), [0, 1, 2, 3], {}).expect
        np.testing.assert_allclose(sx, csx)
        np.testing.assert_allclose(sy, csy)
        np.testing.assert_allclose(sz, csz)

    @pytest.mark.parametrize('method', all_ode_method, ids=all_ode_method)
    def test_sesolver_steping(self, method):
        options = SolverOptions(method=method, atol=1e-7, rtol=1e-8,
                                progress_bar=None)
        solver_obj = SeSolver([self.H1, lambda t, args: args["a"]],
                              args={"a":0.25}, options=options)
        solver_obj.start(basis(2,0), 0)
        sr2 = -(2**.5)/2
        state = solver_obj.step(1)
        np.testing.assert_allclose(expect(sigmax(), state), 0., atol=2e-6)
        np.testing.assert_allclose(expect(sigmay(), state), -1, atol=2e-6)
        np.testing.assert_allclose(expect(sigmaz(), state), 0., atol=2e-6)
        state = solver_obj.step(2, args={"a":0.125})
        np.testing.assert_allclose(expect(sigmax(), state), 0., atol=2e-6)
        np.testing.assert_allclose(expect(sigmay(), state), sr2, atol=2e-6)
        np.testing.assert_allclose(expect(sigmaz(), state), sr2, atol=2e-6)
