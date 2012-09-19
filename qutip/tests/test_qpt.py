# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import os
from numpy.testing import assert_

from qutip import *

def test_qpt_snot():
    "quantum process tomography for snot gate"

    U_psi = snot()
    U_rho = spre(U_psi) * spost(U_psi.dag())
    N = 1
    op_basis = [[qeye(2), sigmax(), 1j*sigmay(), sigmaz()] for i in range(N)]
    op_label = [["i", "x", "y", "z"] for i in range(N)]
    chi1 = qpt(U_rho, op_basis)

    chi2 = zeros((2**(2*N), 2**(2*N)), dtype=complex)
    chi2[1,1] = chi2[1,3] = chi2[3,1] = chi2[3,3] = 0.5

    assert_(norm(chi2-chi1) < 1e-8)


def test_qpt_cnot():
    "quantum process tomography for cnot gate"

    U_psi = cnot()
    U_rho = spre(U_psi) * spost(U_psi.dag())
    N = 2
    op_basis = [[qeye(2), sigmax(), 1j*sigmay(), sigmaz()] for i in range(N)]
    op_label = [["i", "x", "y", "z"] for i in range(N)]
    chi1 = qpt(U_rho, op_basis)

    chi2 = zeros((2**(2*N), 2**(2*N)), dtype=complex)
    chi2[0,0] = chi2[0,1] = chi2[1,0] = chi2[1,1] = 0.25

    chi2[12,0] = chi2[12,1] =  0.25
    chi2[13,0] = chi2[13,1] = -0.25

    chi2[0,12] = chi2[1,12] =  0.25
    chi2[0,13] = chi2[1,13] = -0.25

    chi2[12,12] = chi2[13,13] =  0.25
    chi2[13,12] = chi2[12,13] = -0.25

    assert_(norm(chi2-chi1) < 1e-8)
    
if __name__ == "__main__":
    run_module_suite()
