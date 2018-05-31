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

import numpy as np
from numpy.testing import assert_, run_module_suite

# disable the progress bar
import os

from qutip import sigmax, sigmay, sigmaz, qeye
from qutip import basis, expect
from qutip.interpolate import Cubic_Spline
from qutip import sesolve
from qutip.solver import Options

os.environ['QUTIP_GRAPHICS'] = "NO"


class TestSESolve:
    """
    A test class for the QuTiP Schrodinger Eq. solver
    """

    def check_evolution(self, H, delta, psi0, tlist, analytic_func,
                        U0=None, td_args={}, tol=5e-3):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """

        if U0 is None:
            output = sesolve(H, psi0, tlist, [sigmax(), sigmay(), sigmaz()],
                            args=td_args)
            sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
        else:
            output = sesolve(H, U0, tlist, args=td_args)
            sx = [expect(sigmax(), U*psi0) for U in output.states]
            sy = [expect(sigmay(), U*psi0) for U in output.states]
            sz = [expect(sigmaz(), U*psi0) for U in output.states]

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = np.array([-np.sin(delta*analytic_func(t, td_args))
                                for t in tlist])
        sz_analytic = np.array([np.cos(delta*analytic_func(t, td_args))
                                for t in tlist])

        assert_(max(abs(sx - sx_analytic)) < tol,
                msg="expect X not matching analytic")
        assert_(max(abs(sy - sy_analytic)) < tol,
                msg="expect Y not matching analytic")
        assert_(max(abs(sz - sz_analytic)) < tol,
                msg="expect Z not matching analytic")

    def test_01_1_state_with_const_H(self):
        "sesolve: state with const H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        analytic_func = lambda t, args: t

        self.check_evolution(H1, delta, psi0, tlist, analytic_func)

    def test_01_1_unitary_with_const_H(self):
        "sesolve: unitary operator with const H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        analytic_func = lambda t, args: t

        self.check_evolution(H1, delta, psi0, tlist, analytic_func, U0)

    def test_02_1_state_with_func_H(self):
        "sesolve: state with td func H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        h1_func = lambda t, args: H1*np.exp(-args['alpha']*t)
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(h1_func, delta, psi0, tlist, analytic_func,
                             td_args=td_args)

    def test_02_2_unitary_with_func_H(self):
        "sesolve: unitary operator with td func H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        h1_func = lambda t, args: H1*np.exp(-args['alpha']*t)
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(h1_func, delta, psi0, tlist, analytic_func, U0,
                             td_args=td_args)

    def test_03_1_state_with_list_func_H(self):
        "sesolve: state with td list func H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        h1_coeff = lambda t, args: np.exp(-args['alpha']*t)
        H = [[H1, h1_coeff]]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func,
                             td_args=td_args)

    def test_03_2_unitary_with_list_func_H(self):
        "sesolve: unitary operator with td list func H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        h1_coeff = lambda t, args: np.exp(-args['alpha']*t)
        H = [[H1, h1_coeff]]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func, U0,
                             td_args=td_args)

    def test_04_1_state_with_list_str_H(self):
        "sesolve: state with td list str H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        H = [[H1, 'exp(-alpha*t)']]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func,
                             td_args=td_args)

    def test_04_2_unitary_with_list_func_H(self):
        "sesolve: unitary operator with td list str H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        H = [[H1, 'exp(-alpha*t)']]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func, U0,
                             td_args=td_args)


    def test_05_1_state_with_interp_H(self):
        "sesolve: state with td interp H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        tcub = np.linspace(0, 20, 50)
        S = Cubic_Spline(0, 20, np.exp(-alpha*tcub))
        H = [[H1, S]]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func,
                             td_args=td_args)

    def test_05_2_unitary_with_interp_H(self):
        "sesolve: unitary operator with td interp H"
        delta = 1.0 * 2*np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        H1 = 0.5*delta*sigmax()      # Hamiltonian operator
        tlist = np.linspace(0, 20, 200)

        alpha = 0.1
        td_args = {'alpha':alpha}
        tcub = np.linspace(0, 20, 50)
        S = Cubic_Spline(0, 20, np.exp(-alpha*tcub))
        H = [[H1, S]]
        analytic_func = lambda t, args: ((1 - np.exp(-args['alpha']*t))
                                        /args['alpha'])

        self.check_evolution(H, delta, psi0, tlist, analytic_func, U0,
                             td_args=td_args)

    def compare_evolution(self, H, psi0, tlist,
                        normalize=False, td_args={}, tol=5e-5):
        """
        Compare integrated evolution of unitary operator with state evo
        """
        U0 = qeye(2)
        options = Options(store_states=True, normalize_output=normalize)
        out_s = sesolve(H, psi0, tlist, [sigmax(), sigmay(), sigmaz()],
                        options=options,args=td_args)
        xs, ys, zs = out_s.expect[0], out_s.expect[1], out_s.expect[2]

        out_u = sesolve(H, U0, tlist, options=options, args=td_args)
        xu = [expect(sigmax(), U*psi0) for U in out_u.states]
        yu = [expect(sigmay(), U*psi0) for U in out_u.states]
        zu = [expect(sigmaz(), U*psi0) for U in out_u.states]

        if normalize:
            msg_ext = ". (Normalized)"
        else:
            msg_ext = ". (Not normalized)"
        assert_(max(abs(xs - xu)) < tol,
                msg="expect X not matching" + msg_ext)
        assert_(max(abs(ys - yu)) < tol,
                msg="expect Y not matching" + msg_ext)
        assert_(max(abs(zs - zu)) < tol,
                msg="expect Z not matching" + msg_ext)

    def test_06_1_compare_state_and_unitary_const(self):
        "sesolve: compare state and unitary operator evo - const H"
        eps = 0.2 * 2*np.pi
        delta = 1.0 * 2*np.pi   # atom frequency
        w0 = 0.5*eps
        w1 = 0.5*delta
        H0 = w0*sigmaz()
        H1 = w1*sigmax()
        H = H0 + H1

        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 20, 200)

        self.compare_evolution(H, psi0, tlist,
                        normalize=False, tol=5e-5)
        self.compare_evolution(H, psi0, tlist,
                        normalize=True, tol=5e-5)

    def test_06_2_compare_state_and_unitary_func(self):
        "sesolve: compare state and unitary operator evo - func td"
        eps = 0.2 * 2*np.pi
        delta = 1.0 * 2*np.pi   # atom frequency
        w0 = 0.5*eps
        w1 = 0.5*delta
        H0 = w0*sigmaz()
        H1 = w1*sigmax()
        a = 0.1
        alpha = 0.1
        td_args = {'a':a, 'alpha':alpha}
        H_func = lambda t, args: a*t*H0 + H1*np.exp(-alpha*t)
        H = H_func

        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 20, 200)

        self.compare_evolution(H, psi0, tlist,
                        normalize=False, td_args=td_args, tol=5e-5)
        self.compare_evolution(H, psi0, tlist,
                        normalize=True, td_args=td_args, tol=5e-5)

    def test_06_3_compare_state_and_unitary_list_func(self):
        "sesolve: compare state and unitary operator evo - list func td"
        eps = 0.2 * 2*np.pi
        delta = 1.0 * 2*np.pi   # atom frequency
        w0 = 0.5*eps
        w1 = 0.5*delta
        H0 = w0*sigmaz()
        H1 = w1*sigmax()
        a = 0.1
        w_a = w0
        td_args = {'a':a, 'w_a':w_a}
        h0_func = lambda t, args: a*t
        h1_func = lambda t, args: np.cos(w_a*t)
        H = [[H0, h0_func], [H1, h1_func]]

        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 20, 200)

        self.compare_evolution(H, psi0, tlist,
                        normalize=False, td_args=td_args, tol=5e-5)
        self.compare_evolution(H, psi0, tlist,
                        normalize=True, td_args=td_args, tol=5e-5)

    def test_06_4_compare_state_and_unitary_list_str(self):
        "sesolve: compare state and unitary operator evo - list str td"
        eps = 0.2 * 2*np.pi
        delta = 1.0 * 2*np.pi   # atom frequency
        w0 = 0.5*eps
        w1 = 0.5*delta
        H0 = w0*sigmaz()
        H1 = w1*sigmax()
        w_a = w0

        td_args = {'w_a':w_a}
        H = [H0, [H1, 'cos(w_a*t)']]

        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 20, 200)

        self.compare_evolution(H, psi0, tlist,
                        normalize=False, td_args=td_args, tol=5e-5)
        self.compare_evolution(H, psi0, tlist,
                        normalize=True, td_args=td_args, tol=5e-5)

if __name__ == "__main__":
    run_module_suite()
