# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import numpy as np
from numpy.testing import assert_, run_module_suite
from qutip import *


class TestFloquet:
    """
    A test class for the QuTiP functions for Floquet formalism.
    """

    def testFloquetUnitary(self):
        """
        Floquet: test unitary evolution of time-dependent two-level system
        """

        delta = 1.0 * 2 * pi
        eps0 = 1.0 * 2 * pi
        A = 0.5 * 2 * pi
        omega = sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: sin(args['w'] * t)]]
        e_ops = [num(2)]

        # solve schrodinger equation with floquet solver
        sol = fsesolve(H, psi0, tlist, e_ops, T, args)

        # compare with results from standard schrodinger equation
        sol_ref = mesolve(H, psi0, tlist, [], e_ops, args)

        assert_(max(abs(sol.expect[0] - sol_ref.expect[0])) < 1e-4)


if __name__ == "__main__":
    run_module_suite()
